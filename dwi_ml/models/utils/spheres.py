# -*- coding: utf-8 -*-
import numpy as np
import torch
from dipy.core.sphere import HemiSphere, Sphere


class TorchSphere:
    def __init__(self, dipy_sphere: Sphere, device=None):
        self.sphere = dipy_sphere
        self.vertices = torch.as_tensor(self.sphere.vertices,
                                        dtype=torch.float32,
                                        device=device)

    def find_closest(self, xyz):
        """
        Returns vertices by index of cosine distance to a direction vector
        """
        # Note: See self.sphere.find_closest(), but here we want to work with
        # torch methods.
        #       cos_sim = np.dot(self.vertices, xyz)
        #       return np.argmax(cos_sim)

        # First send the vertices on the right device, i.e. same as target
        # directions
        self.vertices = self.vertices.to(device=xyz.device)

        # We will use cosine similarity to find nearest vertex
        cosine_similarity = torch.matmul(xyz, self.vertices.t())

        # Ordering by similarity. On the last dimension = per time step per
        # sequence in the batch.
        index = torch.argmax(cosine_similarity, dim=-1).type(torch.int16)

        return index


def send_targets_to_half_sphere(t, sphere):
    # toDo. See how to include this to use classification on the half sphere.
    half_sphere = HemiSphere.from_sphere(sphere)
    peak = half_sphere.find_closest(t)

    # Checking cosine similarity:
    if abs(np.dot(peak, t)) > abs(np.dot(peak, -t)):
        t = -t

    return t
