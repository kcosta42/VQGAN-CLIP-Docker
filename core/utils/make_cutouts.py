import torch
import torch.nn as nn

import kornia.augmentation as K

CUTOUTS = {
    'Ji': K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.05, p=0.5),
    'Sh': K.RandomSharpness(sharpness=0.4, p=0.7),
    'Gn': K.RandomGaussianNoise(mean=0.0, std=1., p=0.5),
    'Pe': K.RandomPerspective(distortion_scale=0.7, p=0.7),
    'Ro': K.RandomRotation(degrees=15, p=0.7),
    'Af': K.RandomAffine(degrees=15, translate=0.1, p=0.7, padding_mode='border'),
    'Et': K.RandomElasticTransform(p=0.7),
    'Ts': K.RandomThinPlateSpline(scale=0.3, same_on_batch=False, p=0.7),
    'Er': K.RandomErasing((.1, .4), (.3, 1 / .3), same_on_batch=True, p=0.7),
}


class MakeCutouts(nn.Module):
    def __init__(self, augments, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

        augment_list = []
        for item in augments:
            if item == 'Cr':
                aug = K.RandomCrop(size=(self.cut_size, self.cut_size), p=0.5),
            elif item == 'Re':
                aug = K.RandomResizedCrop(size=(self.cut_size, self.cut_size), scale=(0.1, 1), ratio=(0.75, 1.333), cropping_mode='resample', p=0.5)
            else:
                aug = CUTOUTS[item]
            augment_list.append(aug)

        print(f"Augmentations: {augment_list}")

        self.augs = nn.Sequential(*augment_list)

        self.noise_fac = 0.1

        # Pooling
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def forward(self, input):
        cutouts = []

        for _ in range(self.cutn):
            # Use Pooling
            cutout = (self.av_pool(input) + self.max_pool(input)) / 2
            cutouts.append(cutout)

        batch = self.augs(torch.cat(cutouts, dim=0))

        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch
