import numpy as np
import torch.utils.data
import torchvision.transforms

import voxel
import render


class Cuboid(torch.utils.data.Dataset):
    """Dataset for segmentation."""

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
        self.num_images = num_images
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
        self.generated_images = [
            self._create_cuboid_set()
            for _ in range(num_images)
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
            )
        )

    def _create_cuboid_set(self):
        cube = voxel.create.cuboid(self.side_range, self.space_size)

        angles = np.random.uniform(0, 2 * np.pi, size=3)

        rotation_matrix = voxel.transform.create_rotation_matrix(*angles)

        rotated_cube = voxel.transform.apply_transform(
            cube, rotation_matrix, centred=True, bg_col=0.0
        )

        tensor_cube = (rotated_cube.astype(np.float32) / 255.0)[np.newaxis, ...]

        silhouette = render.silhouette(tensor_cube.squeeze()).astype(np.float32)

        return (
            silhouette,
            (
                silhouette,
                cube,
            ),
        )

    def __len__(self):
        return len(self.generated_images)
