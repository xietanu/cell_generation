import pytest
import torch
import numpy as np

from tests.voxgrid.examples import get_example_0c_voxgrid
from voxgrid import get_alpha, get_values


@pytest.mark.parametrize(
    "voxgrid, expected",
    [
        (
            get_example_0c_voxgrid(as_tensor=False),
            get_example_0c_voxgrid(as_tensor=False),
        ),
        (
            get_example_0c_voxgrid(as_tensor=True),
            get_example_0c_voxgrid(as_tensor=True).squeeze(),
        ),
    ],
)
def test_get_alpha(voxgrid, expected):
    assert np.allclose(get_alpha(voxgrid), expected)
