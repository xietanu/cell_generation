import itertools

import numpy as np

OFFSETS = []

for x, y in itertools.product(range(-2, 3), repeat=2):
    if x == 0 and y == 0:
        continue
    OFFSETS.append([x, y])

OFFSETS = np.array(OFFSETS, dtype=np.float32)

NORMALS = np.array([[*(offset / np.linalg.norm(offset)), 0] for offset in OFFSETS])

VERT_NORMAL = np.array([0, 0, 1], dtype=np.float32)


def create_normal_map(depth_map: np.ndarray) -> np.ndarray:
    if depth_map.ndim != 2:
        raise ValueError("Depth map must be 2D")

    normal_map = np.zeros((*depth_map.shape, 3), dtype=np.float32)
    normal_norm = np.zeros(depth_map.shape, dtype=np.float32)

    for offset, normal in zip(OFFSETS, NORMALS):
        offset_depth_map = np.roll(depth_map, offset.astype(np.int32), axis=(0, 1))
        difference_map = (
            2
            * (offset_depth_map - depth_map - np.min(depth_map))
            / (np.max(depth_map) - np.min(depth_map))
        )
        difference_map[offset_depth_map == 0] = 0
        difference_map[depth_map == 0] = 0
        normal_map += (
            np.cos(np.pi / 2 * difference_map)[:, :, np.newaxis] * VERT_NORMAL
            + np.sin(np.pi / 2 * difference_map)[:, :, np.newaxis] * normal
        )

        normal_norm += np.sign(np.abs(difference_map))

    normal_norm[normal_norm == 0] = 1

    normal_map[:, :, 2] = normal_map[:, :, 2] / normal_norm

    normal_map = normal_map / np.linalg.norm(normal_map, axis=2)[:, :, np.newaxis]

    normal_map[depth_map == 0] = [0, 0, 0]

    return normal_map
