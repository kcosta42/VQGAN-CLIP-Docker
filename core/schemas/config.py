import os

from core.clip.clip import available_models

from typing import List
from dataclasses import dataclass, field


INIT_NOISES = ['', 'gradient', 'pixels']
OPTIMIZERS = ['Adam', 'AdamW', 'Adagrad', 'Adamax', 'DiffGrad', 'AdamP', 'RAdam']
AUGMENTS = ['Ji', 'Sh', 'Gn', 'Pe', 'Ro', 'Af', 'Et', 'Ts', 'Cr', 'Er', 'Re']


@dataclass
class Config:
    prompts: List[str] = field(default_factory=lambda: [])
    image_prompts: List[str] = field(default_factory=lambda: [])
    max_iterations: int = 500
    save_freq: int = 50
    size: List[int] = field(default_factory=lambda: [256, 256])
    init_image: str = ""
    init_noise: str = "gradient"
    init_weight: float = 0.0
    output_dir: str = "/outputs"
    models_dir: str = "/models"
    clip_model: str = 'ViT-B/32'
    vqgan_checkpoint: str = '/models/vqgan_imagenet_f16_16384.ckpt'
    vqgan_config: str = '/configs/models/vqgan_imagenet_f16_16384.json'
    noise_prompt_seeds: List[int] = field(default_factory=lambda: [])
    noise_prompt_weights: List[float] = field(default_factory=lambda: [])
    step_size: float = 0.1
    cutn: int = 32
    cut_pow: float = 1.0
    seed: int = -1
    optimizer: str = 'Adam'
    nwarm_restarts: int = -1
    augments: List[str] = field(default_factory=lambda: ['Af', 'Pe', 'Ji', 'Er'])

    def __post_init__(self):
        if self.init_noise not in INIT_NOISES:
            exit(f"ERROR: \"init_noise\": {self.init_noise}, <-- Noise algorithm not found.\n"
                 f"Currently only the following values are supported: {INIT_NOISES}.")

        if self.optimizer not in OPTIMIZERS:
            exit(f"ERROR: \"optimizer\": {self.optimizer}, <-- Optimizer not found.\n"
                 f"Currently only the following values are supported: {OPTIMIZERS}.")

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/steps", exist_ok=True)
        print(f"Saving outputs in '{self.output_dir}'")

        models = available_models()
        if not os.path.exists(self.clip_model) and self.clip_model not in models:
            exit(f"ERROR: \"clip_model\": {self.clip_model}, <-- Model not found.\n"
                 f"Make sure it is a valid path to a downloaded model or match one of {models}.")

        if not os.path.exists(self.vqgan_config):
            exit(f"ERROR: \"vqgan_config\": {self.vqgan_config}, <-- Configuration file not found.\n"
                 f"Make sure the path is correct (Multiple config files are available in the `./configs/models` directory).")

        if not os.path.exists(self.vqgan_checkpoint):
            exit(f"ERROR: \"vqgan_checkpoint\": {self.vqgan_checkpoint}, <-- Model not found.\n"
                 f"Make sure the path is correct and that you have downloaded the model (Refer to the README).")

    def __str__(self):
        _str = (
            f"Config:\n"
            f"  - prompts: {self.prompts}\n"
            f"  - image_prompts: {self.image_prompts}\n"
            f"  - max_iterations: {self.max_iterations}\n"
            f"  - save_freq: {self.save_freq}\n"
            f"  - size: {self.size}\n"
            f"  - init_image: {self.init_image}\n"
            f"  - init_noise: {self.init_noise}\n"
            f"  - init_weight: {self.init_weight}\n"
            f"  - output_dir: {self.output_dir}\n"
            f"  - models_dir: {self.models_dir}\n"
            f"  - clip_model: {self.clip_model}\n"
            f"  - vqgan_checkpoint: {self.vqgan_checkpoint}\n"
            f"  - vqgan_config: {self.vqgan_config}\n"
            f"  - noise_prompt_seeds: {self.noise_prompt_seeds}\n"
            f"  - noise_prompt_weights: {self.noise_prompt_weights}\n"
            f"  - step_size: {self.step_size}\n"
            f"  - cutn: {self.cutn}\n"
            f"  - cut_pow: {self.cut_pow}\n"
            f"  - seed: {self.seed}\n"
            f"  - optimizer: {self.optimizer}\n"
            f"  - augments: {self.augments}\n"
        )
        return _str
