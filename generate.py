import os
import json
import argparse

import torch
import torch.nn.functional as F

import torchvision.transforms.functional as TF

import numpy as np

from PIL import Image, PngImagePlugin

from tqdm import tqdm

from core.schemas import Config
from core.taming.models import vqgan
from core.clip import clip

from core.utils import MakeCutouts, Normalize, resize_image, get_optimizer
from core.utils.noises import random_noise_image, random_gradient_image
from core.utils.prompt import Prompt, parse_prompt
from core.utils.gradients import ClampWithGrad, vector_quantize


PARAMS: Config = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NORMALIZE = Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                      std=[0.26862954, 0.26130258, 0.27577711], device=DEVICE)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="/configs/generate.json")
    return parser.parse_args()


def load_vqgan_model(config_path, checkpoint_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    model = vqgan.VQModel(**config["params"])
    model.eval().requires_grad_(False)
    model.init_from_ckpt(checkpoint_path)

    del model.loss
    return model


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

    pMs = []
    for prompt in PARAMS.prompts:
        txt, weight, stop = parse_prompt(prompt)
        embed = perceptor.encode_text(clip.tokenize(txt).to(DEVICE)).float()
        pMs.append(Prompt(embed, weight, stop).to(DEVICE))

    for prompt in PARAMS.image_prompts:
        path, weight, stop = parse_prompt(prompt)
        img = Image.open(path)
        pil_image = img.convert('RGB')
        img = resize_image(pil_image, (sideX, sideY))
        batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(DEVICE))
        embed = perceptor.encode_image(NORMALIZE(batch)).float()
        pMs.append(Prompt(embed, weight, stop).to(DEVICE))

    for seed, weight in zip(PARAMS.noise_prompt_seeds, PARAMS.noise_prompt_weights):
        gen = torch.Generator().manual_seed(seed)
        embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
        pMs.append(Prompt(embed, weight).to(DEVICE))

    return pMs


def debug_log(seed):
    print('Using seed:', seed)
    print('Using device:', DEVICE)
    print('Optimising using:', PARAMS.optimizer)
    print('Using text prompts:', PARAMS.prompts)
    print('Using image prompts:', PARAMS.image_prompts)
    print('Using initial image:', PARAMS.init_image)
    print('Noise prompt weights:', PARAMS.noise_prompt_weights)


def synth(z, model):
    z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
    return ClampWithGrad.apply(model.decode(z_q).add(1).div(2), 0, 1)


@torch.no_grad()
def checkin(z, model, i, losses):
    losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
    tqdm.write(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
    out = synth(z, model)
    info = PngImagePlugin.PngInfo()
    info.add_text('comment', f'{PARAMS.prompts}')
    TF.to_pil_image(out[0].cpu()).save(PARAMS.output, pnginfo=info)


def ascend_txt(pMs, model, perceptor, make_cutouts, z, z_orig, i):
    out = synth(z, model)
    iii = perceptor.encode_image(NORMALIZE(make_cutouts(out))).float()

    result = []

    if PARAMS.init_weight:
        result.append(F.mse_loss(z, torch.zeros_like(z_orig)) * ((1 / torch.tensor(i * 2 + 1)) * PARAMS.init_weight) / 2)

    for prompt in pMs:
        result.append(prompt(iii))

    img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:, :, :]
    img = np.transpose(img, (1, 2, 0))
    Image.fromarray(img).save(f"./steps/{i}.png")
    return result


def train(i, optimizer, z, z_min, z_max, pMs, model, perceptor, make_cutouts, z_orig):
    optimizer.zero_grad(set_to_none=True)
    lossAll = ascend_txt(pMs, model, perceptor, make_cutouts, z, z_orig, i)

    if i % PARAMS.display_freq == 0:
        checkin(z, model, i, lossAll)

    loss = sum(lossAll)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        z.copy_(z.maximum(z_min).minimum(z_max))


def main():
    model = load_vqgan_model(PARAMS.vqgan_config, PARAMS.vqgan_checkpoint).to(DEVICE)
    perceptor = clip.load(PARAMS.clip_model, jit=False)[0].eval().requires_grad_(False).to(DEVICE)

    cut_size = perceptor.visual.input_resolution
    make_cutouts = MakeCutouts(PARAMS.augments, cut_size, PARAMS.cutn, cut_pow=PARAMS.cut_pow)

    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]
    z = initialize_image(model)
    z_orig = z.clone()
    z.requires_grad_(True)

    pMs = tokenize(model, perceptor, make_cutouts)
    optimizer = get_optimizer(z, PARAMS.optimizer, PARAMS.step_size)

    seed = PARAMS.seed
    seed = seed if seed else torch.seed()
    torch.manual_seed(seed)

    debug_log(seed)

    i = 0
    try:
        with tqdm() as pbar:
            while True:
                train(i, optimizer, z, z_min, z_max, pMs, model, perceptor, make_cutouts, z_orig)
                if i == PARAMS.max_iterations:
                    break
                i += 1
                pbar.update()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.config):
        exit(f"ERROR: {args.config} not found.")

    with open(args.config, 'r') as f:
        PARAMS = Config(**json.load(f))

    os.makedirs("./steps", exist_ok=True)
    main()
