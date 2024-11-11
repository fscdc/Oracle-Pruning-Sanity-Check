from importlib import import_module
import os, shutil
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import random
import math
from typing import Tuple
from torch import Tensor
from torchvision.transforms import functional as F

class RandomMixUp(torch.nn.Module):
    """Randomly apply MixUp to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()

        if num_classes < 1:
            raise ValueError(
                f"Please provide a valid positive value for the num_classes. Got num_classes={num_classes}"
            )

        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on mixup paper, page 3.
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s

class RandomCutMix(torch.nn.Module):
    """Randomly apply CutMix to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    <https://arxiv.org/abs/1905.04899>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for cutmix.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("Please provide a valid positive value for the num_classes.")
        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        _, H, W = F.get_dimensions(batch)

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s

def get_module(use_v2):
    if use_v2:
        import torchvision.transforms.v2

        return torchvision.transforms.v2
    else:
        import torchvision.transforms

        return torchvision.transforms

def get_mixup_cutmix(*, mixup_alpha, cutmix_alpha, num_categories, use_v2):
    transforms_module = get_module(use_v2)

    mixup_cutmix = []
    if mixup_alpha > 0:
        mixup_cutmix.append(
            transforms_module.MixUp(alpha=mixup_alpha, num_categories=num_categories)
            if use_v2
            else RandomMixUp(num_classes=num_categories, p=1.0, alpha=mixup_alpha)
        )
    if cutmix_alpha > 0:
        mixup_cutmix.append(
            transforms_module.CutMix(alpha=mixup_alpha, num_categories=num_categories)
            if use_v2
            else RandomCutMix(num_classes=num_categories, p=1.0, alpha=mixup_alpha)
        )
    if not mixup_cutmix:
        return None

    return transforms_module.RandomChoice(mixup_cutmix)

class Data(object):
    def __init__(self, args):
        self.args = args

        train_folder, val_folder = "train", "val"
        kwargs = {}
        if args.dataset in [
            "imagenet",
            "imagenet_subset_100",
            "imagenet_subset_200",
        ]:
            kwargs = {
                "train_folder": train_folder,
                "val_folder": val_folder,
            }
        
        # Set up train set and test set
        dataset = import_module("data.%s" % args.dataset)
        dataset_path = os.path.join(args.data_path, args.dataset_dir)
        train_set, test_set = dataset.get_dataset(dataset_path, **kwargs)
        train_subset = dataset.get_sub_trainset(dataset_path, **kwargs)

        train_sampler = None
        train_subset_sampler = None

        if args.distributed:
            train_sampler = data.distributed.DistributedSampler(train_set)
            train_subset_sampler = data.distributed.DistributedSampler(train_subset)

        # set up collate_fn
        num_classes = num_classes_dict[args.dataset]
        mixup_cutmix = get_mixup_cutmix(
            mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, num_categories=num_classes, use_v2=args.use_v2
        )
        if mixup_cutmix is not None:
            def collate_fn(batch):
                return mixup_cutmix(*default_collate(batch))
        else:
            collate_fn = default_collate

        self.train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=args.workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        self.train_loader_nocollate = DataLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=args.workers,
            shuffle=True,
            pin_memory=True,
        )

        self.train_subset_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            shuffle=False,
            pin_memory=True,
        )        

        self.test_loader = DataLoader(
            test_set,
            batch_size=256,
            num_workers=args.workers,
            shuffle=False,
            pin_memory=True,
        )

        self.train_set = train_set
        self.test_set = test_set
        self.train_subset = train_subset
        self.train_sampler = train_sampler
        self.train_subset_sampler = train_subset_sampler
        self.train_loader = self.train_loader
        self.train_loader_nocollate = self.train_loader_nocollate
        self.train_subset_loader = self.train_subset_loader
        self.test_loader = self.test_loader

num_classes_dict = {
    "imagenet": 1000,
    "imagenet_subset_100": 100,
    "imagenet_subset_200": 200,
    "tinyimagenet": 200,
    "imdb": 2,
}

# shape [N, C, H, W]
input_size_dict = {
    "imagenet": (1, 3, 224, 224),
    "imagenet_subset_100": (1, 3, 224, 224),
    "imagenet_subset_200": (1, 3, 224, 224),
    "tinyimagenet": (1, 3, 64, 64),
    "imdb": -1,
}

