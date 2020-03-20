# -*- coding: utf-8 -*-
import logging

import numpy as np
from torch.utils.data import Dataset, Sampler


class BatchSampler(Sampler):
    """Samples sequences using a number of required timesteps, without
    replacement.

    It is also possible to restrict of volumes in a batch, in which case
    a number of "cycles" is also required, which for how many batches the
    same volumes should be re-used before sampling new ones.

    NOTE: Actual batch sizes might be different than `batch_size` depending
    on later data augmentation. This sampler takes streamline cutting and
    resampling into consideration, and as such, will return more (or less)
    points than the provided `batch_size`.

    Arguments:
        data_source : Dataset
            Dataset to sample from.
        batch_size : int
            Number of required points in a batch. This will be approximated as
            the final batch size depends on data augmentation (streamline cutting
            or resampling).
        rng : np.random.RandomState
            Random number generator.
        n_volumes : int
            Optional; maximum number of volumes to be used in a single batch.
            If None, always use all volumes.
        cycles : int
            Optional, but required if `n_volumes` is given.
            Number of batches re-using the same volumes before sampling new ones.
    """

    def __init__(self, data_source: MultiSubjectDataset,  # Says error because it doesn't use super().__init__
                 batch_size: int, rng: np.random.RandomState,
                 n_volumes: int = None, cycles: int = None):
        if not isinstance(data_source, Dataset):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Dataset, but got data_source={}"
                             .format(data_source))
        if (not isinstance(batch_size, int) or isinstance(batch_size, bool)
                or batch_size <= 0):
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if cycles and not n_volumes:
            raise ValueError("If `cycles_per_volume_batch` is defined, "
                             "`n_volumes` should be defined. Got: "
                             "n_volumes={}, cycles={}"
                             .format(n_volumes, cycles))
        self.data_source = data_source
        self.batch_size = batch_size
        self._rng = rng
        self.n_volumes = n_volumes
        self.cycles = cycles

    def __iter__(self):
        """First sample the volumes to be used from a given number of desired
        volumes, then sample streamline ids inside those volumes.

        Returns
        -------
        batch : list of tuple of (relative_streamline_id, tractodata_id)
        """
        global_streamlines_ids = np.arange(len(self.data_source))
        global_streamlines_mask = np.ones_like(global_streamlines_ids,
                                               dtype=np.bool)

        while True:
            # Weight volumes by their number of remaining streamlines
            streamlines_per_volume = np.array(
                [np.sum(global_streamlines_mask[start:end])
                 for tid, (start, end) in
                 self.data_source.subjID_to_streamlineID.items()])

            if np.sum(streamlines_per_volume) == 0:
                logging.info("No streamlines remain for this epoch, "
                             "stopping...")
                break

            if self.n_volumes:
                weights = \
                    streamlines_per_volume / np.sum(streamlines_per_volume)

                # Choose only non-empty volumes
                n_volumes = min(self.n_volumes, np.count_nonzero(weights))
                sampled_tids = self._rng.choice(
                    np.arange(len(self.data_source.data_list)),
                    size=n_volumes, replace=False, p=weights)
            else:
                sampled_tids = self.data_source.subjID_to_streamlineID.keys()
                n_volumes = len(sampled_tids)

            # Compute the number of *original* timesteps required per volume
            # (before resampling)
            n_timesteps_per_volume = self.batch_size / n_volumes

            if self.cycles:
                iterator = range(self.cycles)
            else:
                # Infinite iterator
                iterator = iter(int, 1)

            for _ in iterator:
                # For each volume, randomly choose streamlines that haven't been
                # chosen yet
                batch = []

                for tid in sampled_tids:
                    # Get the global streamline ids corresponding to this volume
                    start, end = self.data_source.subjID_to_streamlineID[tid]
                    volume_global_ids = global_streamlines_ids[start:end]

                    total_volume_timesteps = 0
                    while True:
                        # Filter for available (unmasked) streamlines
                        available_streamline_ids = \
                            volume_global_ids[global_streamlines_mask[start:end]]

                        # No streamlines remain for this volume
                        if len(available_streamline_ids) == 0:
                            break

                        # Sample a batch of streamlines and get their lengths
                        sample_global_ids = \
                            self._rng.choice(available_streamline_ids, 256)
                        sample_timesteps = \
                            self.data_source.streamline_timesteps[sample_global_ids]

                        volume_batch_fulfilled = False
                        # Keep total volume length under the maximum
                        if (total_volume_timesteps + np.sum(sample_timesteps) >
                                n_timesteps_per_volume):
                            # Select only enough streamlines to fill the
                            # required length
                            cumulative_sum = np.cumsum(sample_timesteps)
                            selected_mask = \
                                cumulative_sum < (n_timesteps_per_volume -
                                                  total_volume_timesteps)
                            sample_global_ids = sample_global_ids[selected_mask]
                            sample_timesteps = sample_timesteps[selected_mask]
                            volume_batch_fulfilled = True

                        # Add this streamline's length to total length
                        total_volume_timesteps += np.sum(sample_timesteps)

                        # Mask the sampled streamline
                        global_streamlines_mask[sample_global_ids] = 0

                        # Fetch tractodata relative id
                        sample_relative_ids = sample_global_ids - start

                        # Add sample to batch
                        for sample_id in sample_relative_ids:
                            batch.append((sample_id, tid))

                        if volume_batch_fulfilled:
                            break

                if len(batch) == 0:
                    logging.info("No more streamlines remain in any of the "
                                 "selected volumes! Moving to new cycle!")
                    break

                yield batch
