from core.taming.modules.losses.lpips import LPIPS
from core.taming.modules.losses.vqperceptual import VQLPIPSWithDiscriminator, DummyLoss

__all__ = [
    LPIPS,
    DummyLoss,
    VQLPIPSWithDiscriminator
]
