"""Dataset for general generation from images, where images are cached."""
import os

import torch.utils.data
import torchvision.transforms

import cv2
import numpy as np


class CachedImages(torch.utils.data.Dataset):
    """Dataset for general generation from images, where images are cached."""

    def __init__(
        self,
        image_folder: str,
        image_names: np.ndarray,
        *,
        train=False,
        grayscale=False,
        crop: float = 1,
        size: tuple[int, int] = (32, 32),
    ):
        self.img_folder = image_folder
        self.train = train
        self.grayscale = grayscale
        self.crop = crop
        self.size = size
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
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
                torchvision.transforms.Lambda(
                    lambda x: 2 * (x - x.min()) / (x.max() - x.min()) - 1
                ),
            ]
        )
        self.images = image_names
        self.cached_images = [self.get_image(img_name) for img_name in self.images]

    def __getitem__(self, index):
        if self.train:
            transform = self.train_transform
        else:
            transform = self.transform

        mask = self.cached_images[index]

        return (
            transform(mask),
            self.transform(mask),
        )

    def get_image(self, name):
        mask = cv2.imread(
            os.path.join(self.img_folder, name),
            cv2.IMREAD_GRAYSCALE if self.grayscale else cv2.IMREAD_COLOR,
        )

        top = int((1 - self.crop) * mask.shape[0] / 2)
        height = mask.shape[0] - 2 * top
        left = int((1 - self.crop) * mask.shape[1] / 2)
        width = mask.shape[1] - 2 * left

        mask = mask[top : top + height, left : left + width]

        if self.size != (mask.shape[0], mask.shape[1]):
            mask = cv2.resize(mask, self.size)

        return mask

    def __len__(self):
        return len(self.cached_images)


def split_folder(folder: str, train_ratio: float, max_images: int | None = None):
    image_list = os.listdir(folder)

    if max_images is not None:
        image_list = image_list[:max_images]

    np.random.shuffle(image_list)

    train_size = int(len(image_list) * train_ratio)
    train_list = np.array(image_list[:train_size])
    test_list = np.array(image_list[train_size:])
    return train_list, test_list
