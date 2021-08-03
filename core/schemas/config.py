from typing import List
from dataclasses import dataclass, field


@dataclass
class Config:
    prompts: List[str] = field(default_factory=lambda: [])
    image_prompts: List[str] = field(default_factory=lambda: [])
    max_iterations: int = 500
    display_freq: int = 50
    size: List[int] = field(default_factory=lambda: [256, 256])
    init_image: str = ""
    init_noise: str = "gradient"
    init_weight: float = 0.0
    clip_model: str = '/models/ViT-B-32.pt'
    vqgan_checkpoint: str = '/config/vqgan_imagenet_f16_16384.ckpt'
    vqgan_config: str = '/config/vqgan_imagenet_f16_16384.json'
    noise_prompt_seeds: List[int] = field(default_factory=lambda: [])
    noise_prompt_weights: List[float] = field(default_factory=lambda: [])
    step_size: float = 0.1
    cutn: int = 32
    cut_pow: float = 1.0
    seed: int = 42
    optimizer: str = 'Adam'
    output: str = 'output.png'
    augments: List[str] = field(default_factory=lambda: ['Af', 'Pe', 'Ji', 'Er'])
