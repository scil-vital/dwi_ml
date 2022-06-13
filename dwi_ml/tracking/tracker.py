# -*- coding: utf-8 -*-
import logging
import multiprocessing
import os

from scilpy.image.datasets import DataVolume
from scilpy.tracking.tracker import Tracker as ScilpyTracker

from dwi_ml.tracking.propagator import DWIMLPropagator
from dwi_ml.tracking.seed import DWIMLSeedGenerator

logger = logging.getLogger('tracker_logger')


class DWIMLTracker(ScilpyTracker):
    def __init__(self, propagator: DWIMLPropagator, mask: DataVolume,
                 seed_generator: DWIMLSeedGenerator, nbr_seeds, min_nbr_pts,
                 max_nbr_pts, max_invalid_dirs, compression_th=0.1,
                 nbr_processes=1, save_seeds=False, rng_seed=1234,
                 track_forward_only=False, simultanenous_tracking=False,
                 device=None, log_level=logging.WARNING):
        """
        Parameters: See scilpy.
        ----------
        propagator: now a dwi_ml version of the propagator.
        use_gpu: bool
            New option to use multi-tracking. Not ready
        hdf_file:
            Necessary to open a new hdf handle a at process.
        """

        # Warning about the mask being an MRIData instead of DataVolume but
        # ok! Modified to be able to use as tensor more easily for torch.
        mmap_mode = None  # Not used here, we deal with hdf5.
        super().__init__(propagator, mask, seed_generator, nbr_seeds,
                         min_nbr_pts, max_nbr_pts, max_invalid_dirs,
                         compression_th, nbr_processes, save_seeds, mmap_mode,
                         rng_seed, track_forward_only)

        if nbr_processes > 2 and not propagator.dataset.is_lazy:
            raise ValueError("Multiprocessing only works with lazy data.")

        # Set device
        self.simultanenous_tracking = simultanenous_tracking
        self.device = device

        # Sending model and data to device.
        self.propagator.move_to(device=self.device)

        logger.setLevel(log_level)

        # Increase the printing frequency as compared to super
        self.printing_frequency = 5

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
            return super().track()

    def _prepare_multiprocessing_pool(self, tmpdir=None):
        """
        Prepare multiprocessing pool. We do not need to deal with data
        management, contrary to scilpy.
        """
        # Clearing data from memory
        self.propagator.reset_data(reload_data=False)

        pool = multiprocessing.Pool(self.nbr_processes)
        return pool

    def _reload_data_for_new_process(self, init_args):
        """Nothing to do here. hdf5 deals well with multi-processes."""

        logger.info("Preparing for process id {}".format(os.getpid()))
        self.propagator.reset_data(reload_data=True)

    def simultanenous_tracking_on_gpu(self):
        """
        Creating all seeds at once and propagating all streamlines together.
        """
        # random_generator, indices = self.seed_generator.init_generator(
        #    self.rng_seed, self.skip)
        # seeds = self.seed_generator.get_next_n_pos(
        #    random_generator, indices, self.skip)

        # toDo Finish preparing code to use an equivalent of
        #  _get_line_both_directions but working on many streamlines at once.
        #  See Philippe's code in vitalabai.
        raise NotImplementedError
