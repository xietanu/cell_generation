import numpy as np
import torch.utils.data
import torchvision.transforms

import voxel
import render


class Blob(torch.utils.data.Dataset):
    """Dataset for segmentation."""

    def __init__(
        self,
        blob_size_range: tuple[int, int],
        n_blob_range: tuple[int, int],
        space_size: tuple[int, int, int],
        num_images: int,
        train: bool = False,
    ):
        self.blob_size_range = blob_size_range
        self.n_blob_range = n_blob_range
        self.img_size = space_size[:2]
        self.num_images = num_images
        self.space_size = space_size
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
        self.generated_images = [self._create_blob_set() for _ in range(num_images)]

    def __getitem__(self, index):
        if self.train:
            transform = self.train_transform
        else:
            transform = self.transform
        return (
            transform(self.generated_images[index][0]),
            (
                transform(self.generated_images[index][0]),
                self.generated_images[index][1],
            ),
        )

    def _create_blob_set(self):
        n_blob = np.random.randint(*self.n_blob_range)

        cube = voxel.create.blob(self.space_size, n_blob, self.blob_size_range)

        angles = np.random.uniform(0, 2 * np.pi, size=3)

        rotation_matrix = voxel.transform.create_rotation_matrix(*angles)

        rotated_cube = voxel.transform.apply_transform(
            cube, rotation_matrix, centred=True
        )

        silhouette = render.silhouette(rotated_cube).astype(np.float32)

        return (
            silhouette,
            cube,
        )

    def __len__(self):
        return len(self.generated_images)
