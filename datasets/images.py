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
    ):
        self.img_folder = image_folder
        self.train = train
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(
                    lambda x: 2 * (x - x.min()) / (x.max() - x.min()) - 1
                ),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                # torchvision.transforms.RandomRotation(180, fill=-1),
            ]
        )
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
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
            os.path.join(self.img_folder, self.images[index]), cv2.IMREAD_GRAYSCALE
        )

        return (
            transform(mask),
            self.transform(mask),
        )

    def __len__(self):
        return len(self.images)
