# -*- coding: utf-8 -*-
import logging

import numpy as np
import torch

from dwi_ml.data.processing.streamlines.data_augmentation import \
    resample_or_compress
from dwi_ml.models.main_models import MainModelOneInput, MainModelAbstract, ModelWithDirectionGetter
from dwi_ml.testing.utils import prepare_dataset_one_subj

logger = logging.getLogger('tester_logger')


def load_sft_from_hdf5(subj_id, hdf5_file, subset_name,
                       streamline_group):
    # Load SFT as in dataloader, except we don't loop on many subject,
    # we don't verify streamline ids (loading all), and we don't split /
    # reverse streamlines. But we resample / compress.
    subset, subj_idx = prepare_dataset_one_subj(
        hdf5_file, subj_id, subset_name=subset_name,
        streamline_groups=[streamline_group],
        lazy=False, cache_size=1)

    logging.info("Loading its streamlines as SFT.")
    streamline_group_idx = subset.streamline_groups.index(streamline_group)
    subj_data = subset.subjs_data_list.get_subj_with_handle(subj_idx)
    subj_sft_data = subj_data.sft_data_list[streamline_group_idx]
    sft = subj_sft_data.as_sft()

    return sft


class Tester:
    """
    Similar to the trainer, it loads the data, runs the model and gets the
    loss.

    However, the reference streamlines are loaded from a given SFT rather than
    from the hdf5. This choice allows to test the loss on various bundles for
    a better interpretation of the models' performances.
    """
    def __init__(self, model: MainModelAbstract,
                 subj_id, hdf5_file, subset_name,
                 batch_size: int = None, device: torch.device = None):
        """
        Parameters
        ----------
        model: MainModelAbstract
        subj_id: str
        hdf5_file: str
        subset_name: str
        batch_size: int
        device: torch.Device
        """
        self.device = device
        self.model = model
        self.model.eval()  # Removes dropout.
        self.model.move_to(device)
        self.batch_size = batch_size

        self.experiment_params = None
        self.input_info = None

        # Load subject
        logging.debug("Loading subject {} from hdf5.".format(subj_id))
        self.subset, self.subj_idx = prepare_dataset_one_subj(
            hdf5_file, subj_id, subset_name=subset_name,
            volume_groups=self._volume_groups, streamline_groups=[],
            lazy=False, cache_size=1, log_level=logging.WARNING)

    @property
    def _volume_groups(self):
        return None

    def run_model_on_sft(self, sft, compute_loss=False):
        """
        Equivalent of one validation pass.

        Parameters
        ----------
        sft: StatefulTractogram
        compute_loss: bool
            If True, compute the loss per streamline.
            (If False, the method returns the outputs after a forward pass
            only.)

        Returns
        -------
        outputs:
            - Gaussian model: outputs = ([], [])
            - Fisher Von mises: not Implemented
            - Other: []
        losses:
        """
        sft = resample_or_compress(sft, self.model.step_size,
                                   self.model.compress_lines)
        sft.to_vox()
        sft.to_corner()
        nb_streamlines = len(sft)

        # Verify the batch size
        batch_size = self.batch_size or nb_streamlines
        nb_batches = int(np.ceil(nb_streamlines / batch_size))
        logging.info("Preparing to run the model on {} streamlines with "
                     "batches of {} streamlines = {} batches."
                     .format(nb_streamlines, batch_size, nb_batches))

        # Prepare output formats.
        outputs = None
        losses = []
        mean_per_line = None

        # Run all batches
        batch_start = 0
        batch_end = batch_size
        with torch.no_grad():
            for batch in range(nb_batches):
                logging.info("  Batch #{}:  {} - {}"
                             .format(batch + 1, batch_start, batch_end - 1))

                # 1. Prepare batch. Same process as in trainer, but no option
                # to add noise.
                streamlines = [
                    torch.as_tensor(s, dtype=torch.float32, device=self.device)
                    for s in sft.streamlines[batch_start:batch_end]]
                if not self.model.direction_getter.add_eos:
                    # We don't use the last coord because it does not have an
                    # associated target direction.
                    streamlines_f = [s[:-1, :] for s in streamlines]
                else:
                    streamlines_f = streamlines
                inputs = self._prepare_inputs(streamlines_f)

                # 2. Run forward
                batch_out = self.model(inputs, streamlines_f)
                outputs = self.model.merge_batches_outputs(outputs, batch_out)

                # 3. Compute loss: not averaged = one tensor of losses per
                # streamline.
                if compute_loss:
                    if isinstance(self.model, ModelWithDirectionGetter):
                        tmp_losses = self.model.compute_loss(
                            batch_out, streamlines, average_results=False)
                    else:
                        tmp_losses = self.model.compute_loss(
                            batch_out, streamlines)
                    losses.extend(tmp_losses)

                # Prepare next batch
                batch_start = batch_end
                batch_end = min(batch_start + batch_size, len(sft))

        if compute_loss:
            losses = [loss.cpu().numpy() for loss in losses]
            mean_per_line = np.asarray([np.mean(loss) for loss in losses])
            print("Final losses for the {} streamlines is : {:.4f} ± {:.4f}. "
                  "Max: {:.4f}. Min: {:.4f}"
                  .format(len(sft), np.mean(mean_per_line),
                          np.std(mean_per_line), np.max(mean_per_line),
                          np.min(mean_per_line)))

        return sft, outputs, losses, mean_per_line

    def _prepare_inputs(self, streamlines):
        return None


class TesterOneInput(Tester):
    model: MainModelOneInput

    def __init__(self, volume_group, *args, **kw):
        self.volume_group = volume_group
        super().__init__(*args, **kw)
        self.input_group_idx = self.subset.volume_groups.index(volume_group)

    @property
    def _volume_groups(self):
        return [self.volume_group]

    def _prepare_inputs(self, streamlines):
        return self.model.prepare_batch_one_input(
            streamlines, self.subset, self.subj_idx, self.input_group_idx)
