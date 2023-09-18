import numpy as np
import torch.utils.data
import torchvision.transforms

import voxel
import render


class Spheroid(torch.utils.data.Dataset):
    """Dataset for segmentation."""

    def __init__(
        self,
        length_range: tuple[int, int],
        r_range: tuple[int, int],
        space_size: tuple[int, int, int],
        num_images: int,
        *,
        provide_silhouette: bool = True,
        provide_angles: bool = False,
        provide_voxel: bool = False,
    ):
        self.length_range = length_range
        self.r_range = r_range
        self.img_size = space_size[:2]
        self.num_images = num_images
        self.space_size = space_size
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(
                    lambda x: 2 * (x - x.min()) / (x.max() - x.min()) - 1
                ),
            ]
        )
        self.generated_images = [
            self._create_cuboid_set(
                provide_silhouette=provide_silhouette,
                provide_angles=provide_angles,
                provide_voxel=provide_voxel,
            )
            for _ in range(num_images)
        ]

    def __getitem__(self, index):
        return self.generated_images[index]

    def _create_cuboid_set(self, *, provide_silhouette, provide_angles, provide_voxel):
        length_multi = np.random.uniform(*self.length_range)
        r = np.random.uniform(*self.r_range)

        cube = voxel.create.spheroid(length_multi, r, self.space_size)

        angles = np.random.uniform(0, 2 * np.pi, size=3)

        rotation_matrix = voxel.transform.create_rotation_matrix(*angles)

        rotated_cube = voxel.transform.apply_transform(
            cube, rotation_matrix, centred=True
        )

        tensor_cube = (rotated_cube.astype(np.float32) / 255.0)[np.newaxis, ...]

        silhouette = render.silhouette(tensor_cube.squeeze()).astype(np.float32)

        silhouette = self.transform(silhouette)

        inputs = []
        if provide_silhouette:
            inputs.append(silhouette)
        if provide_angles:
            inputs.append(torch.Tensor(angles))
        if provide_voxel:
            inputs.append(tensor_cube)

        if len(inputs) == 1:
            inputs = inputs[0]
        else:
            inputs = tuple(inputs)

        return (
            inputs,
            (
                silhouette,
                angles,
                tensor_cube,
            ),
        )

    def __len__(self):
        return len(self.generated_images)
