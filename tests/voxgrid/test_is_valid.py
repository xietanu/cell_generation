import pytest
import torch
import numpy as np

from voxgrid import is_valid_voxel_grid


@pytest.mark.parametrize(
    "voxgrid",
    [
        np.random.uniform(0, 1, (3, 3, 3)),
        np.random.uniform(0, 1, (3, 3, 3, 1)),
        np.random.uniform(0, 1, (3, 3, 3, 2)),
        np.random.uniform(0, 1, (3, 3, 3, 4)),
        np.random.uniform(0, 1, (5, 3, 3, 3, 1)),
        np.random.uniform(0, 1, (5, 3, 3, 3, 2)),
        np.random.uniform(0, 1, (5, 3, 3, 3, 4)),
        torch.rand((1, 3, 3, 3)),
        torch.rand((2, 3, 3, 3)),
        torch.rand((4, 3, 3, 3)),
        torch.rand((5, 1, 3, 3, 3)),
        torch.rand((5, 2, 3, 3, 3)),
        torch.rand((5, 4, 3, 3, 3)),
    ],
)
def test_is_valid_voxel_grid_should_be_valid(voxgrid):
    assert is_valid_voxel_grid(voxgrid)

@pytest.mark.parametrize(
    "voxgrid",
    [
        np.ones((3, 3, 3)) * 2,
        torch.ones((1, 3, 3, 3)) * 2,
        np.zeros((5, 3, 3, 3, 2)) - 0.1,
        torch.zeros((5, 3, 3, 3, 2)) - 0.1,
    ],
)
def test_is_valid_voxel_grid_not_normed(voxgrid):
    assert not is_valid_voxel_grid(voxgrid)

@pytest.mark.parametrize(
    "voxgrid",
    [
        np.random.uniform(0, 1, (2, 3, 3, 3)),
        np.random.uniform(0, 1, (5, 2, 3, 3, 3)),
        torch.rand((3, 3, 3, 2)),
        torch.rand((5, 3, 3, 3, 2)),
    ],
)
def test_is_valid_voxel_grid_wrong_channel_loc(voxgrid):
    assert not is_valid_voxel_grid(voxgrid)

@pytest.mark.parametrize(
    "voxgrid",
    [
        np.random.uniform(0, 1, (2, 3, 3, 3, 3, 3)),
        np.random.uniform(0, 1, (3, 3)),
        torch.rand((3, 3, 3)),
        torch.rand((3, 3)),
        torch.rand((5, 2, 3, 3, 3, 3)),
    ],
)
def test_is_valid_voxel_grid_wrong_n_dims(voxgrid):
    assert not is_valid_voxel_grid(voxgrid)

@pytest.mark.parametrize(
    "voxgrid",
    [
        np.random.uniform(0, 1, (3, 3, 3, 7)),
        np.random.uniform(0, 1, (5, 3, 3, 3, 3)),
        torch.rand((3, 3, 3, 3)),
        torch.rand((5, 6, 3, 3, 3)),
    ],
)
def test_is_valid_voxel_grid_wrong_n_channels(voxgrid):
    assert not is_valid_voxel_grid(voxgrid)