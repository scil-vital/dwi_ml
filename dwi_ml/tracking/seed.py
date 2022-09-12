# -*- coding: utf-8 -*-
from dipy.io.stateful_tractogram import Space, Origin
from scilpy.tracking.seed import SeedGenerator


class DWIMLSeedGenerator(SeedGenerator):
    """
    Seed generator with added methods to generate many seeds instead of one
    at the time, for GPU processing of many streamlines at once.
    """
    def __init__(self, data, voxres):
        # # torch trilinear interpolation uses origin='corner', space=vox.
        super().__init__(data, voxres,
                         space=Space.VOX, origin=Origin('corner'))

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

        len_seeds = len(self.seeds_vox)

        if len_seeds == 0:
            return []

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
            x, y, z = self.seeds_vox[indices[inds[i]]]

            seed = [x + r_x[i], y + r_y[i], z + r_z[i]]

            if self.space == Space.VOXMM:
                # Should not happen now. Kept in case we modify something.
                # Also, this equation is only true in corner.
                seed *= self.voxres
            elif self.space != Space.VOX:
                raise NotImplementedError("Not ready for rasmm")

            seeds.append(seed)

        return seeds
