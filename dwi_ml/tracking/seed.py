# -*- coding: utf-8 -*-
import numpy as np
import torch
from dipy.io.stateful_tractogram import Space

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

        # CONTRARY TO SCILPY, WE USE VOX SPACE (we keep corner origin)
        self.space = Space.VOX

    def move_to(self, device):
        self.data = self.data.to(device=device)
        self.device = device

    def get_next_n_pos(self, random_generator, indices, which_seeds):
        """
        Generate the next n seed positions (Space=voxmm, origin=corner).
        Heavy, should be used on GPU.

        Parameters
        ----------
        random_generator : numpy random generator
            Initialized numpy number generator.
        indices : numpy array
            Indices of current seeding map.
        which_seeds : numpy array
            Seed numbers (i.e. IDs) to be processed.

        Return
        ------
        seed_pos: List[tuple]
            Positions of next seeds expressed in VOXEL SPACE.
        """
        # todo Bring this to torch and use correct device?
        #   We would need to change the seed generator.

        # todo This copies scilpy's, but see Charles's GPU version, uses dipy
        #  random_seeds_from_mask
        len_seeds = len(self.seeds)
        if len_seeds == 0:
            return []

        voxel_dim = np.asarray(self.voxres)

        # Voxel selection from the seeding mask
        inds = which_seeds % len_seeds

        # Sub-voxel initial positioning
        # Prepare sub-voxel random movement now (faster out of loop)
        n = len(which_seeds)
        r_x = random_generator.uniform(0, 1, size=n)
        r_y = random_generator.uniform(0, 1, size=n)
        r_z = random_generator.uniform(0, 1, size=n)

        seeds = []
        for i in range(len(which_seeds)):
            x, y, z = self.seeds[indices[inds[i]]]

            if self.space == Space.VOXMM:
                # Should not happen now. Kept in case we modify something.
                seeds.append((x * self.voxres[0] + r_x[i] * voxel_dim[0],
                              y * self.voxres[1] + r_y[i] * voxel_dim[1],
                              z * self.voxres[2] + r_z[i] * voxel_dim[2]))
            elif self.space == Space.VOX:
                seeds.append((x + r_x[i],
                              y + r_y[i],
                              z + r_z[i]))
            else:
                raise NotImplementedError("Not ready for rasmm")
        return seeds
