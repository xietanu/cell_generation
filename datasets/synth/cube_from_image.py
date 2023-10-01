"""Dataset for testing colour voxel generation."""
import numpy as np
import torch.utils.data
import torchvision.transforms

import voxgrid


class CubeFromImage(torch.utils.data.Dataset):
    """Dataset for testing colour voxel generation."""

    def __init__(
        self,
        image_path: str,
        grey: bool,
        space_size: tuple[int, int, int],
        num_images: int,
        *,
        train: bool = False,
    ):
        self.image_path = image_path
        self.grey = grey
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
        cube = voxgrid.create.cube_from_image(
            self.image_path, self.space_size, self.grey
        )

        image = cube.rotated(*np.random.uniform(0, 2 * np.pi, size=3)).create_image()

        return (
            image.as_array(),
            (
                image.as_array(),
                cube.as_array(),
            ),
        )

    def __len__(self):
        return len(self.generated_images)
