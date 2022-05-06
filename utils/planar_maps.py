"""
Classes for creating simple ring manifolds.
"""

import numpy as np
from sklearn.utils.validation import check_random_state



class AffineMap:

    def transform(self, theta):
        """
        Returns (n x 3) array of neural encoding of a vector
        of positions (theta).
        """
        coords = np.column_stack([
            np.cos(theta),
            np.sin(theta)
        ])
        return coords @ np.transpose(self.W) + self.b[None, :]


class PlanarNdMap(AffineMap):

    def __init__(self, W, b):
        self.W = W  # shape == (n, 2)
        self.b = b  # shape == (n,)


class Planar3dMap(AffineMap):

    def __init__(self, roll, pitch, yaw, radius, center):
        """
        Initializes encoding of position.
        """

        # Roll - Pitch - Yaw matrices.
        R = np.array([
            [1,           0,              0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll),  np.cos(roll)],
        ])
        P = np.array([
            [np.cos(pitch),  0, np.sin(pitch)],
            [            0,  1,             0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ])
        Y = np.array([
            [np.cos(yaw), -np.sin(yaw),  0],
            [np.sin(yaw),  np.cos(yaw),  0],
            [          0,            0,  1],
        ])

        # Embedding matrix from 2d -> 3d.
        E = np.array(
            [[1, 0], [0, 1], [0, 0]]
        ) * radius

        # By convention: roll, then pitch, then yaw
        self.W = R @ P @ Y @ E  # (3 x 2) matrix
        self.b = center
