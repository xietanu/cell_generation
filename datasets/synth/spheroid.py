"""Dataset for spheroid generation."""
import numpy as np
import torch.utils.data
import torchvision.transforms

import voxgrid


class Spheroid(torch.utils.data.Dataset):
    """Dataset for spheroid generation."""

    def __init__(
        self,
        r_range: tuple[int, int],
        length_multiplier_range: tuple[float, float],
        space_size: tuple[int, int, int],
        num_images: int,
        *,
        train: bool = False,
    ):
        self.r_range = r_range
        self.length_multiplier_range = length_multiplier_range
        self.space_size = space_size
        self.train = train
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(
                    lambda x: 2 * (x - x.min()) / (x.max() - x.min()) - 1
                ),
            ]
        )
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
        self.generated_images = [self._create_spheroid_set() for _ in range(num_images)]

    def __getitem__(self, index):
        if self.train:
            transform = self.train_transform
        else:
            transform = self.transform

        silh, outputs = self.generated_images[index]
        return (
            transform(silh),
            (
                transform(outputs[0]),
                outputs[1],
            ),
        )

    def _create_spheroid_set(self):
        length_multiplier = np.random.uniform(*self.length_multiplier_range)
        r = int(np.random.uniform(self.r_range[0], self.r_range[1]) / length_multiplier)
    

        cuboid = voxgrid.create.spheroid(length_multiplier, r, self.space_size)  # type: ignore

        mask = cuboid.create_image(*np.random.uniform(0, 2 * np.pi, size=3))

        return (
            mask.as_array(),
            (
                mask.as_array(),
                cuboid.as_array(),
            ),
        )

    def __len__(self):
        return len(self.generated_images)
