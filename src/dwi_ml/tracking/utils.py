# -*- coding: utf-8 -*-
import logging


def prepare_step_size_vox(step_size, res):
    if step_size:
        # Our step size is in voxel space = we require isometric voxels.
        if not res[0] == res[1] == res[2]:
            raise ValueError("We are expecting isometric voxels.")

        step_size_vox_space = step_size / res[0]
        normalize_directions = True
        logging.info("Step size in voxel space will be {}"
                     .format(step_size_vox_space))
    else:
        # Step size 0 is understood here as 'keep model output as is', i.e.
        # multiplying by 1 works.
        step_size_vox_space = 1
        normalize_directions = False

    return step_size_vox_space, normalize_directions

