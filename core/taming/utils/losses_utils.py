import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import VGG
from torchvision.models.vgg import load_state_dict_from_url

from typing import List, Union, cast


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def load_vgg(model_dir: str, pretrained: bool = False, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False

    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    model = VGG(make_layers(cfg, batch_norm=False), **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/vgg16-397923af.pth',
                                              model_dir=model_dir,
                                              file_name="vgg16-397923af.pth",
                                              progress=True)
        model.load_state_dict(state_dict)
        print(f"Loaded pretrained VGG16 model from '{model_dir}/vgg16-397923af.pth'")

    return model
