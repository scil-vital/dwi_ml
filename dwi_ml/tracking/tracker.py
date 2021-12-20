# -*- coding: utf-8 -*-
from scilpy.image.datasets import DataVolume
from scilpy.tracking.tracker import Tracker as ScilpyTracker

from dwi_ml.tracking.propagator import DWIMLPropagator
from dwi_ml.tracking.seed import DWIMLSeedGenerator


class DWIMLTracker(ScilpyTracker):
    def __init__(self, propagator: DWIMLPropagator, mask: DataVolume,
                 seed_generator: DWIMLSeedGenerator, nbr_seeds, min_nbr_pts,
                 max_nbr_pts, max_invalid_dirs, compression_th=0.1,
                 nbr_processes=1, save_seeds=False, mmap_mode=None,
                 rng_seed=1234, track_forward_only=False,
                 simultanenous_tracking=False, device=None):
        """
        Parameters: See scilpy.
        ----------
        propagator: now a dwi_ml version of the propagator.
        use_gpu: added option
        """

        # Warning about the mask being an MRIData instead of DataVolume but
        # ok! Modified to be able to use as tensor more easily for torch.
        super().__init__(propagator, mask, seed_generator, nbr_seeds,
                         min_nbr_pts, max_nbr_pts, max_invalid_dirs,
                         compression_th, nbr_processes, save_seeds, mmap_mode,
                         rng_seed, track_forward_only)

        # Set device
        self.simultanenous_tracking = simultanenous_tracking
        self.device = device

        # Sending model and data to device.
        self.propagator.move_to(device=self.device)

    def track(self):
        """
        Scilpy's Tracker.track():
            - calls _get_streamlines, or
            - deals with parallel sub-processes and calls _get_streamlines_sub,
              which call _get_streamlines for each process.

        Here adding the GPU usage. Other changes for dwi_ml will be reflected
        in _get_streamlines.
        """
        if self.simultanenous_tracking:
            self.simultanenous_tracking_on_gpu()
        else:
            # On CPU, with possibility of parallel processing.
            super().track()

    def simultanenous_tracking_on_gpu(self):
        """
        Creating all seeds at once and propagating all streamlines together.
        """
        random_generator, indices = self.seed_generator.init_generator(
            self.rng_seed, self.skip)
        seeds = self.seed_generator.get_next_n_pos(
            random_generator, indices, self.skip)

        # toDo Finish preparing code to use an equivalent of
        #  _get_line_both_directions but working on many streamlines at once.
        #  See Philippe's code in vitalabai.
        raise NotImplementedError
