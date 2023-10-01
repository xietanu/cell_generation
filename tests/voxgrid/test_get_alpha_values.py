import pytest
import numpy as np

from tests.voxgrid.examples import get_example_1_channel_voxgrid
from voxgrid import get_alpha, get_values


@pytest.mark.parametrize(
    "voxgrid, expected",
    [
        (
            get_example_1_channel_voxgrid().array,
            get_example_1_channel_voxgrid().alpha[np.newaxis, ..., np.newaxis].copy(),
        ),
        (
            get_example_1_channel_voxgrid().tensor.squeeze(),
            get_example_1_channel_voxgrid().alpha,
        ),
        (
            get_example_1_channel_voxgrid().get_array_without_channels(),
            get_example_1_channel_voxgrid().alpha,
        ),
        (
            get_example_1_channel_voxgrid().get_array_with_channels(),
            get_example_1_channel_voxgrid().alpha[..., np.newaxis].copy(),
        )
    ],
)
def test_get_alpha(voxgrid, expected):
    print(voxgrid.shape, expected.shape)
    print(voxgrid.squeeze())
    print(expected.squeeze())
    assert np.allclose(get_alpha(voxgrid), expected)
