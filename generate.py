import os
import json
import argparse

import torch

from core.schemas import Config
from core.taming.models import vqgan


PARAMS: Config = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="/configs/generate.json")
    return parser.parse_args()


def check_nodels_integrity():
    os.makedirs("./steps", exist_ok=True)

    if not os.path.exists(PARAMS.clip_model):
        # TODO: Download model
        exit(f"ERROR: {PARAMS.clip_model} not found.")

    if not os.path.exists(PARAMS.vqgan_config):
        # TODO: Download config
        exit(f"ERROR: {PARAMS.vqgan_config} not found.")

    if not os.path.exists(PARAMS.vqgan_checkpoint):
        # TODO: Download model
        exit(f"ERROR: {PARAMS.vqgan_checkpoint} not found.")


def load_vqgan_model(config_path, checkpoint_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    model = vqgan.VQModel(**config["params"])

    model.eval().requires_grad_(False)

    model.init_from_ckpt(checkpoint_path)
    del model.loss

    return model


def main():
    check_nodels_integrity()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_vqgan_model(PARAMS.vqgan_config, PARAMS.vqgan_checkpoint).to(device)


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.config):
        exit(f"ERROR: {args.config} not found.")

    with open(args.config, 'r') as f:
        PARAMS = Config(**json.load(f))

    main()
