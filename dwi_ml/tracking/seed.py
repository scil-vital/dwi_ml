# -*- coding: utf-8 -*-
import numpy as np
import torch

from scilpy.tracking.seed import SeedGenerator


class DWIMLSeedGenerator(SeedGenerator):
    """
    Seed generator with added methods to generate many seeds instead of one
    at the time, for GPU processing of many streamlines at once.
    """

    def __init__(self, data, voxres, device=None):
        super().__init__(data, voxres)

        self.data = torch.tensor(self.data)
        self.device = device
        if device is not None:
            self.move_to(device)

    def move_to(self, device):
        self.data.to(device=device)
        self.device = device

    def get_next_n_pos(self, random_generator, indices, which_seeds):
        """
        Generate the next n seed positions (Space=voxmm, origin=corner).
        Heavy, should be used on GPU.

        Parameters
        ----------
        random_generator : numpy random generator
            Initialized numpy number generator.
        indices : List
            Indices of current seeding map.
        which_seeds : List[int]
            Seed numbers (i.e. IDs) to be processed.

        Return
        ------
        seed_pos: List[tuple]
            Positions of next seeds expressed in mm.
        """
        len_seeds = len(self.seeds)
        if len_seeds == 0:
            return []

        voxel_dim = np.asarray(self.voxres)

        # Voxel selection from the seeding mask
        inds = torch.fmod(which_seeds, len_seeds)
        x, y, z = self.seeds[indices[inds]]

        # Sub-voxel initial positioning
        n = len(which_seeds)
        r_x = random_generator.uniform(0, voxel_dim[0], size=n)
        r_y = random_generator.uniform(0, voxel_dim[1], size=n)
        r_z = random_generator.uniform(0, voxel_dim[2], size=n)

        return x * self.voxres[0] + r_x, y * self.voxres[1] \
            + r_y, z * self.voxres[2] + r_z
