from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ExampleVoxelGrid:
    array: np.ndarray
    tensor: torch.Tensor
    values: np.ndarray
    alpha: np.ndarray

    def get_array_without_channels(self) -> np.ndarray:
        return self.array[0, :, :, :, -1]

    def get_array_with_channels(self) -> np.ndarray:
        return self.array[0, :, :, :, :]

    def get_array_with_n(self) -> np.ndarray:
        return self.array

    def get_tensor_with_channels(self) -> torch.Tensor:
        return self.tensor[0, :, :, :, :]

    def get_tensor_with_n(self) -> torch.Tensor:
        return self.tensor


def get_example_1_channel_voxgrid():
    voxels = np.zeros((3, 3, 3), dtype=np.float32)
    voxels[0, 0, 0] = 1
    voxels[0, 1, 0] = 1
    voxels[0, 1, 1] = 1
    voxels[1, 1, 1] = 0.5
    voxels[1, 0, 2] = 1

    array = voxels[np.newaxis, ..., np.newaxis].copy()
    tensor = torch.from_numpy(voxels).unsqueeze(0).unsqueeze(0)
    alpha = voxels.copy()
    values = np.ones_like(alpha)

    return ExampleVoxelGrid(array, tensor, values, alpha)
