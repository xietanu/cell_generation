import numpy as np

import render
import render.utils


def lit(
    voxel_object: np.ndarray, light_angle: float, ambient_light: float = 0.25
) -> np.ndarray:
    if voxel_object.ndim != 3:
        raise ValueError("Normal map must be 3D")

    normal_map = render.utils.create_normal_map(render.depth_map(voxel_object))

    rendered = np.zeros(normal_map.shape[:2], dtype=np.float32)

    angles = (
        np.arctan2(normal_map[:, :, 1], normal_map[:, :, 0])
        + light_angle
        + 3 * np.pi / 2
    )

    angle_intensity = np.cos(angles)

    vert_intensity = np.linalg.norm(normal_map[:, :, :2], axis=2)

    rendered = (angle_intensity * vert_intensity + 1 + ambient_light) / (
        2 + ambient_light
    )

    rendered[render.utils.create_background_mask(voxel_object)] = 0

    rendered = (np.clip(rendered, 0, 1) * 255).astype(np.uint8)

    return rendered
