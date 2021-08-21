import os
import json
import argparse

import torch
from torch.utils.data import DataLoader

import torchvision.transforms.functional as TF
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

from tqdm import tqdm

from core.schemas import TrainConfig
from core.utils import global_seed
from core.utils.loader import safe_load
from core.taming.models import vqgan


PARAMS: TrainConfig = None
DEVICE = torch.device(os.environ.get("DEVICE", 'cuda' if torch.cuda.is_available() else 'cpu'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to configuration file.")
    return parser.parse_args()


def save_model(model, optimizers, epoch, path):
    save_dict = {
        "epoch": epoch,
        "global_step": model.global_step,
        "state_dict": model.state_dict(),
        "optimizer_states": [
            optimizers[0].state_dict(),
            optimizers[1].state_dict(),
        ]
    }
    torch.save(save_dict, path)
    tqdm.write(f"Checkpoint saved in {path}")


def main():
    dataset = ImageFolder(PARAMS.data_dir, T.Compose(
        [
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(PARAMS.params["embed_dim"]),
            T.CenterCrop(PARAMS.params["embed_dim"]),
            T.ToTensor()
        ]
    ))
    loader = DataLoader(dataset, PARAMS.batch_size, shuffle=True)

    PARAMS.params["model_dir"] = PARAMS.models_dir
    model = vqgan.VQModel(**PARAMS.params).to(DEVICE)
    model.learning_rate = PARAMS.batch_size * PARAMS.base_learning_rate
    model.global_step = 0

    optimizers, _ = model.configure_optimizers()
    epoch = 0

    if PARAMS.resume_checkpoint:
        save_dict = safe_load(PARAMS.resume_checkpoint, map_location='cpu')
        epoch = save_dict["epoch"]
        model.global_step = save_dict["global_step"]
        model.load_state_dict(save_dict["state_dict"])
        optimizers[0].load_state_dict(save_dict["optimizer_states"][0])
        optimizers[1].load_state_dict(save_dict["optimizer_states"][1])
        print(f"Restored model from {PARAMS.resume_checkpoint}")

    while epoch < PARAMS.epochs:
        for i, (images, _) in tqdm(enumerate(loader), total=len(loader)):
            images.to(DEVICE)

            losses = []
            for j, opt in enumerate(optimizers):
                loss = model.training_step(images, i, j, device=DEVICE)
                losses.append(loss.item())

                opt.zero_grad()
                loss.backward()

                opt.step()

            tqdm.write(f"Epoch: {epoch} | Batch: {i} | losses: {losses}")

            if i % 1000 == 0:
                save_model(model, optimizers, epoch, f"{PARAMS.models_dir}/checkpoints/last.ckpt")

                with torch.no_grad():
                    dec, _ = model(model.get_input(images, device=DEVICE))
                    TF.to_pil_image(dec[0].cpu()).save(f"{PARAMS.output_dir}/training/{epoch}_{i}.png")

            model.global_step += 1
        epoch += 1

    save_model(model, optimizers, epoch, f"{PARAMS.models_dir}/checkpoints/final.ckpt")


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.config):
        exit(f"ERROR: {args.config} not found.")

    print(f"Loading configuration from '{args.config}'")
    with open(args.config, 'r') as f:
        PARAMS = TrainConfig(**json.load(f))

    print(f"Running on {DEVICE}.")
    print(PARAMS)

    global_seed(PARAMS.seed)

    main()
