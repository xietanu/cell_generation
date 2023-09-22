import pytest
import torch
import numpy as np

from voxgrid import get_n_channels


@pytest.mark.parametrize(
    "voxgrid, expected",
    [
        (np.random.uniform(0, 1, (3, 3, 3)), 0),
        (np.random.uniform(0, 1, (3, 3, 3, 1)), 1),
        (np.random.uniform(0, 1, (3, 3, 3, 2)), 2),
        (np.random.uniform(0, 1, (3, 3, 3, 4)), 4),
        (np.random.uniform(0, 1, (5, 3, 3, 3, 1)), 1),
        (np.random.uniform(0, 1, (5, 3, 3, 3, 2)), 2),
        (np.random.uniform(0, 1, (5, 3, 3, 3, 4)), 4),
        (torch.rand((1, 3, 3, 3)), 1),
        (torch.rand((2, 3, 3, 3)), 2),
        (torch.rand((4, 3, 3, 3)), 4),
        (torch.rand((5, 1, 3, 3, 3)), 1),
        (torch.rand((5, 2, 3, 3, 3)), 2),
        (torch.rand((5, 4, 3, 3, 3)), 4),
    ],
)
def test_get_n_channels(voxgrid, expected):
    assert get_n_channels(voxgrid) == expected
