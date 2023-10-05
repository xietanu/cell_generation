"""Dataset for cuboid generation."""
import numpy as np
import torch.utils.data
import torchvision.transforms

import voxgrid


class Cuboid(torch.utils.data.Dataset):
    """Dataset for cuboid generation."""

    def __init__(
        self,
        side_range: tuple[int, int],
        space_size: tuple[int, int, int],
        num_images: int,
        *,
        train: bool = False,
        alpha: float = 1,
        noise: float = 0,
        occlude: tuple[float, float] = (0, 0),
    ):
        if noise < 0 or noise > 1:
            raise ValueError("noise must be between 0 and 1")

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
            ]
        )
        self.generated_images = [
            self._create_cuboid_set(alpha, noise, occlude) for _ in range(num_images)
        ]

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

    def _create_cuboid_set(
        self, alpha: float = 1, noise: float = 0, occlude: tuple[float, float] = (0, 0)
    ):
        side_lengths = tuple(
            np.random.randint(
                self.side_range[0], self.side_range[1] + 1, size=3, dtype=int
            )
        )

        cuboid = voxgrid.create.cuboid(side_lengths, self.space_size, alpha=alpha)  # type: ignore

        mask = cuboid.rotated(*np.random.uniform(0, 2 * np.pi, size=3)).create_image()
        mask_array = mask.as_array()

        if occlude[1] > 0:
            occlusion = np.sqrt(np.random.uniform(occlude[0], occlude[1]))

            height = int(mask_array.shape[0] * occlusion)
            width = int(mask_array.shape[1] * occlusion)

            top = np.random.randint(0, self.img_size[0] - height)
            left = np.random.randint(0, self.img_size[1] - width)

            mask_array[top : top + height, left : left + width] = 0

        if noise > 0:
            noise_mask = np.random.uniform(-1, 1, size=mask_array.shape)
            mask_array[noise_mask >= 1 - noise] = noise_mask[noise_mask >= 1 - noise]
            mask_array[noise_mask <= -1 + noise] = noise_mask[noise_mask <= -1 + noise]

        return (
            mask_array,
            (
                mask_array,
                cuboid.as_array(),
            ),
        )

    def __len__(self):
        return len(self.generated_images)
