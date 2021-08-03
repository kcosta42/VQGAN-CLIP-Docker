# https://github.com/pratogab/batch-transforms

import torch


class Normalize:
    """Applies the :class:`~torchvision.transforms.Normalize` transform to a batch of images.

    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence):
            Sequence of means for each channel.
        std (sequence):
            Sequence of standard deviations for each channel.
        inplace(bool,optional):
            Bool to make this operation in-place.
        dtype (torch.dtype,optional):
            The data type of tensors to which the transform will be applied.
        device (torch.device,optional):
            The device of tensors to which the transform will be applied.
    """
    def __init__(self, mean, std, inplace=False, dtype=torch.float, device='cpu'):
        self.mean = torch.as_tensor(mean, dtype=dtype, device=device)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype, device=device)[None, :, None, None]
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        tensor.sub_(self.mean).div_(self.std)
        return tensor
