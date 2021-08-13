import os

from dataclasses import dataclass


@dataclass
class TrainConfig:
    base_learning_rate: float = 4.5e-6
    batch_size: int = 1
    epochs: int = 1000
    data_dir: str = "./data"
    output_dir: str = "./outputs"
    models_dir: str = "./models"
    resume_checkpoint: str = ""
    seed: int = -1
    params: dict = None

    def __post_init__(self):
        if not os.path.exists(self.data_dir):
            exit(f"ERROR: \"data_dir\": {self.data_dir}, <-- Data direcotry not found.\n"
                 f"Make sure the path is correct (Follow instructions in the README).")

        ckpt_dir = os.path.join(self.models_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Checkpoints will be saved in {ckpt_dir}")

        train_dir = os.path.join(self.output_dir, "training")
        os.makedirs(train_dir, exist_ok=True)
        print(f"Training outputs will be saved in {train_dir}")

        if self.resume_checkpoint and not os.path.exists(self.resume_checkpoint):
            exit(f"ERROR: \"resume_checkpoint\": {self.resume_checkpoint}, <-- Model not found.\n"
                 f"Make sure the path is correct (Follow instructions in the README).")

    def __str__(self):
        _str = (
            f"Config:\n"
            f"  - base_learning_rate: {self.base_learning_rate}\n"
            f"  - batch_size: {self.batch_size}\n"
            f"  - epochs: {self.epochs}\n"
            f"  - data_dir: {self.data_dir}\n"
            f"  - output_dir: {self.output_dir}\n"
            f"  - models_dir: {self.models_dir}\n"
            f"  - resume_checkpoint: {self.resume_checkpoint}\n"
            f"  - seed: {self.seed}\n"
            f"  - params: {self.params}\n"
        )
        return _str
