"""Dataset for cuboid generation."""
import numpy as np
import torch.utils.data
import torchvision.transforms

import voxel


class Cuboid(torch.utils.data.Dataset):
    """Dataset for cuboid generation."""

    def __init__(
        self,
        side_range: tuple[int, int],
        space_size: tuple[int, int, int],
        num_images: int,
        *,
        train: bool = False,
    ):
        self.side_range = side_range
        self.img_size = space_size[:2]
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
                torchvision.transforms.RandomRotation(180, fill=-1),
            ]
        )
        self.generated_images = [self._create_cuboid_set() for _ in range(num_images)]

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

    def _create_cuboid_set(self):
        side_lengths = tuple(
            np.random.randint(
                self.side_range[0], self.side_range[1] + 1, size=3, dtype=int
            )
        )

        cuboid = voxel.create.cuboid(side_lengths, self.space_size)  # type: ignore

        mask = cuboid.create_mask(*np.random.uniform(0, 2 * np.pi, size=3))

        return (
            mask.as_array(),
            (
                mask.as_array(),
                cuboid.as_array(),
            ),
        )

    def __len__(self):
        return len(self.generated_images)
