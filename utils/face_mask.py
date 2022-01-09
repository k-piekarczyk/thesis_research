# Zero-indexed landmark points making up the outer bounds of the mask
from typing import Any, List, Tuple

import numpy as np


outer_bound_pts = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109
]

'''
Only takes landmarks generated with FaceMesh, returns a list of (x, y) tuples
'''


def get_face_contour(landmarks: Any, width: int, height: int) -> np.array(Tuple[int, int]):
    points = []

    for idx in outer_bound_pts:
        lm = landmarks.landmark[idx]
        points.append((int(lm.x * width), int(lm.y * height)))

    # for lm in landmarks.landmark:
    #     points.append((int(lm.x * width), int(lm.y * height)))
    
    return np.array(points)
