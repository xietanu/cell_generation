import os

import torch.utils.data
import torchvision.transforms

import cv2
import numpy as np


class AutoImgCached(torch.utils.data.Dataset):
    """Dataset for segmentation."""

    def __init__(
        self,
        image_folder: str,
        voxel_folder: str,
        img_range: tuple[int, int],
        train=False,
    ):
        self.img_folder = image_folder
        self.voxel_folder = voxel_folder
        self.train = train
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(
                    lambda x: 2 * (x - x.min()) / (x.max() - x.min()) - 1
                ),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomRotation(180, fill=-1),
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
        self.images = os.listdir(image_folder)[img_range[0] : img_range[1]]
        self.cached_pairs = [self._load_pair(img_name) for img_name in self.images]

    def __getitem__(self, index):
        if self.train:
            transform = self.train_transform
        else:
            transform = self.transform

        mask, voxel = self.cached_pairs[index]

        return (
            transform(mask),
            (
                transform(mask),
                voxel,
            ),
        )

    def _load_pair(self, img_name: str):
        mask = cv2.imread(os.path.join(self.img_folder, img_name), cv2.IMREAD_GRAYSCALE)

        voxel = np.load(
            os.path.join(self.voxel_folder, img_name.split(".")[0] + ".npy")
        )
        return mask, voxel

    def __len__(self):
        return len(self.cached_pairs)
