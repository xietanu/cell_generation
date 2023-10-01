"""Dataset for general generation from images, where images are cached."""
import os

import torch.utils.data
import torchvision.transforms

import cv2
import numpy as np


class Images(torch.utils.data.Dataset):
    """Dataset for general generation from images, where images are cached."""

    def __init__(
        self,
        image_folder: str,
        image_names: np.ndarray,
        train=False,
        grayscale=False,
        crop: float = 0,
        size: tuple[int, int] = (32, 32),
    ):
        self.img_folder = image_folder
        self.train = train
        self.grayscale = grayscale
        self.crop = crop
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(0, 1),
                torchvision.transforms.Resize(size),
                torchvision.transforms.Lambda(
                    lambda x: 2 * (x - x.min()) / (x.max() - x.min()) - 1
                ),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
            ]
        )
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(0, 1),
                torchvision.transforms.Resize(size),
                torchvision.transforms.Lambda(
                    lambda x: 2 * (x - x.min()) / (x.max() - x.min()) - 1
                ),
            ]
        )
        self.images = image_names

    def __getitem__(self, index):
        if self.train:
            transform = self.train_transform
        else:
            transform = self.transform

        mask = cv2.imread(
            os.path.join(self.img_folder, self.images[index]),
            cv2.IMREAD_GRAYSCALE if self.grayscale else cv2.IMREAD_COLOR,
        )

        top = int((1 - self.crop) * mask.shape[0] / 2)
        height = mask.shape[0] - 2 * top
        left = int((1 - self.crop) * mask.shape[1] / 2)
        width = mask.shape[1] - 2 * left

        mask = mask[top : top + height, left : left + width]

        return (
            transform(mask),
            self.transform(mask),
        )

    def __len__(self):
        return len(self.images)
