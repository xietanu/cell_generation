"""Dataset for testing colour voxel generation."""
import numpy as np
import torch.utils.data
import torchvision.transforms  #

import cv2

import voxgrid


class ObjFromImage(torch.utils.data.Dataset):
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

        image = cv2.imread(
            image_path, cv2.IMREAD_GRAYSCALE if grey else cv2.IMREAD_COLOR
        )

        if not grey:
            image = image[:, :, ::-1]

        width = image.shape[1]
        height = width
        depth = image.shape[0] // height

        space = np.zeros((*space_size, 2 if grey else 4), dtype=np.float32)

        left = (space_size[0] - width) // 2
        top = (space_size[1] - height) // 2
        front = (space_size[2] - depth) // 2

        for i in range(depth):
            space[top : top + height, left : left + width, front + i, :-1] = image[
                i * height : (i + 1) * height, :
            ] / 255
            space[top : top + height, left : left + width, front + i, -1] = (
                np.sum(image[i * height : (i + 1) * height, :], axis=-1) > 0
            ).astype(np.int32)

        self.obj = voxgrid.VoxGrid(space)

        self.generated_images = [self._create_image_set() for _ in range(num_images)]

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

    def _create_image_set(self):
        image = self.obj.rotated(
            *np.random.uniform(0, 2 * np.pi, size=3)
        ).create_image()

        return (
            image.as_array(),
            (
                image.as_array(),
                self.obj.as_array(),
            ),
        )

    def __len__(self):
        return len(self.generated_images)
