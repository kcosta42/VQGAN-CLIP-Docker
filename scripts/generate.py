import os
import json
import argparse

import torch
import torch.nn.functional as F

import torchvision.transforms.functional as TF

from PIL import Image

from tqdm import tqdm

from core.schemas import Config
from core.clip import clip

from core.utils import MakeCutouts, Normalize, resize_image, get_optimizer, load_vqgan_model
from core.utils.noises import random_noise_image, random_gradient_image
from core.utils.prompt import Prompt, parse_prompt
from core.utils.gradients import ClampWithGrad, vector_quantize


PARAMS: Config = None
DEVICE = torch.device(os.environ.get("DEVICE", 'cuda' if torch.cuda.is_available() else 'cpu'))
NORMALIZE = Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                      std=[0.26862954, 0.26130258, 0.27577711], device=DEVICE)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to configuration file.")
    return parser.parse_args()


def initialize_image(model):
    f = 2**(model.decoder.num_resolutions - 1)
    toksX, toksY = PARAMS.size[0] // f, PARAMS.size[1] // f
    sideX, sideY = toksX * f, toksY * f

    def encode(img):
        pil_image = img.convert('RGB').resize((sideX, sideY), Image.LANCZOS)
        pil_tensor = TF.to_tensor(pil_image)
        z, *_ = model.encode(pil_tensor.to(DEVICE).unsqueeze(0) * 2 - 1)
        return z

    if PARAMS.init_image:
        z = encode(Image.open(PARAMS.init_image))
    elif PARAMS.init_noise == 'pixels':
        z = encode(random_noise_image(PARAMS.size[0], PARAMS.size[1]))
    elif PARAMS.init_noise == 'gradient':
        z = encode(random_gradient_image(PARAMS.size[0], PARAMS.size[1]))
    else:
        e_dim = model.quantize.e_dim
        n_toks = model.quantize.n_e

        one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=DEVICE), n_toks).float()
        z = one_hot @ model.quantize.embedding.weight
        z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)

    return z


def tokenize(model, perceptor, make_cutouts):
    f = 2**(model.decoder.num_resolutions - 1)
    toksX, toksY = PARAMS.size[0] // f, PARAMS.size[1] // f
    sideX, sideY = toksX * f, toksY * f

    prompts = []
    for prompt in PARAMS.prompts:
        txt, weight, stop = parse_prompt(prompt)
        embed = perceptor.encode_text(clip.tokenize(txt).to(DEVICE)).float()
        prompts.append(Prompt(embed, weight, stop).to(DEVICE))

    for prompt in PARAMS.image_prompts:
        path, weight, stop = parse_prompt(prompt)
        img = Image.open(path)
        pil_image = img.convert('RGB')
        img = resize_image(pil_image, (sideX, sideY))
        batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(DEVICE))
        embed = perceptor.encode_image(NORMALIZE(batch)).float()
        prompts.append(Prompt(embed, weight, stop).to(DEVICE))

    for seed, weight in zip(PARAMS.noise_prompt_seeds, PARAMS.noise_prompt_weights):
        gen = torch.Generator().manual_seed(seed)
        embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
        prompts.append(Prompt(embed, weight).to(DEVICE))

    return prompts


def synth(z, *, model):
    z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
    return ClampWithGrad.apply(model.decode(z_q).add(1).div(2), 0, 1)


@torch.no_grad()
def checkin(z, losses, **kwargs):
    losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
    tqdm.write(f"step: {kwargs['step']}, loss: {sum(losses).item():g}, losses: {losses_str}")
    out = synth(z, model=kwargs['model'])

    filename = f"{PARAMS.output_dir}/{'_'.join(PARAMS.prompts).replace(' ', '_')}.png"
    TF.to_pil_image(out[0].cpu()).save(filename)


def ascend_txt(z, **kwargs):
    out = synth(z, model=kwargs['model'])
    iii = kwargs['perceptor'].encode_image(NORMALIZE(kwargs['make_cutouts'](out))).float()

    step = kwargs['step']
    result = []
    if PARAMS.init_weight:
        result.append(F.mse_loss(z, torch.zeros_like(kwargs['z_orig'])) * ((1 / torch.tensor(step * 2 + 1)) * PARAMS.init_weight) / 2)

    for prompt in kwargs['prompts']:
        result.append(prompt(iii))

    TF.to_pil_image(out[0].cpu()).save(f"{PARAMS.output_dir}/steps/{step}.png")
    return result


def train(z, **kwargs):
    kwargs['optimizer'].zero_grad(set_to_none=True)
    lossAll = ascend_txt(z, **kwargs)

    if kwargs['step'] % PARAMS.save_freq == 0 or kwargs['step'] == PARAMS.max_iterations:
        checkin(z, lossAll, **kwargs)

    loss = sum(lossAll)
    loss.backward()
    kwargs['optimizer'].step()

    with torch.no_grad():
        z.copy_(z.maximum(kwargs['z_min']).minimum(kwargs['z_max']))


def main():
    model = load_vqgan_model(PARAMS.vqgan_config, PARAMS.vqgan_checkpoint, PARAMS.models_dir).to(DEVICE)
    perceptor = clip.load(PARAMS.clip_model, device=DEVICE, root=PARAMS.models_dir)[0].eval().requires_grad_(False).to(DEVICE)

    cut_size = perceptor.visual.input_resolution
    make_cutouts = MakeCutouts(PARAMS.augments, cut_size, PARAMS.cutn, cut_pow=PARAMS.cut_pow)

    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]
    z = initialize_image(model)
    z_orig = z.clone()
    z.requires_grad_(True)

    prompts = tokenize(model, perceptor, make_cutouts)
    optimizer = get_optimizer(z, PARAMS.optimizer, PARAMS.step_size)

    kwargs = {
        'model': model,
        'perceptor': perceptor,
        'optimizer': optimizer,
        'prompts': prompts,
        'make_cutouts': make_cutouts,
        'z_orig': z_orig,
        'z_min': z_min,
        'z_max': z_max,
    }
    try:
        for step in tqdm(range(PARAMS.max_iterations)):
            kwargs['step'] = step + 1
            train(z, **kwargs)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.config):
        exit(f"ERROR: {args.config} not found.")

    print(f"Loading configuration from '{args.config}'")
    with open(args.config, 'r') as f:
        PARAMS = Config(**json.load(f))

    print(f"Running on {DEVICE}.")
    PARAMS.seed = PARAMS.seed if PARAMS.seed != -1 else torch.seed()
    torch.manual_seed(PARAMS.seed)

    print(PARAMS)

    main()
