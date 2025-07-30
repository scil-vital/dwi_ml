# -*- coding: utf-8 -*-
import logging

import numpy as np
import torch
from tqdm import tqdm

from dwi_ml.data.processing.streamlines.data_augmentation import \
    resample_or_compress
from dwi_ml.models.main_models import (MainModelOneInput,
                                       ModelWithDirectionGetter)
from dwi_ml.testing.utils import prepare_dataset_one_subj

logger = logging.getLogger('tester_logger')


def load_sft_from_hdf5(subj_id: str, hdf5_file: str, subset_name: str,
                       streamline_group: str):
    # Load SFT as in dataloader, except we don't loop on many subject,
    # we don't verify streamline ids (loading all), and we don't split /
    # reverse streamlines. But we resample / compress.
    subset = prepare_dataset_one_subj(
        hdf5_file, subj_id, subset_name=subset_name,
        streamline_groups=[streamline_group],
        lazy=False, cache_size=1)

    logging.info("Loading its streamlines as SFT.")
    streamline_group_idx = subset.streamline_groups.index(streamline_group)
    subj_data = subset.subjs_data_list.get_subj_with_handle(subject_idx=0)
    subj_sft_data = subj_data.sft_data_list[streamline_group_idx]
    sft = subj_sft_data.as_sft()

    return sft


class TesterWithDirectionGetter:
    """
    Similar to the trainer, it loads the data, runs the model and gets the
    loss.

    However, the reference streamlines are loaded from a given SFT rather than
    from the hdf5. This choice allows to test the loss on various bundles for
    a better interpretation of the models' performances.
    """
    def __init__(self, model: ModelWithDirectionGetter,
                 subj_id, hdf5_file, subset_name,
                 batch_size: int = None, device: torch.device = None):
        """
        Parameters
        ----------
        model: ModelWithDirectionGetter
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
        logging.info("Loading subject {} from hdf5.".format(subj_id))
        self.subset = prepare_dataset_one_subj(
            hdf5_file, subj_id, subset_name=subset_name,
            volume_groups=self._volume_groups, streamline_groups=[],
            lazy=False, cache_size=1, log_level=logging.WARNING)
        self.subj_idx = 0

    @property
    def _volume_groups(self):
        return None

    def run_model_on_sft(self, sft, compute_loss=False):
        """
        Equivalent of one validation pass.

        Parameters
        ----------
        sft: StatefulTractogram
            Will be resampled / compress based on model's params.
        compute_loss: bool
            If True, compute the loss per streamline.
            (If False, the method returns the outputs after a forward pass
            only.)

        Returns
        -------
        sft: StatefulTractogram
            The tractogram, formatted as required by your model:
            to_vox, to_corner, possibly resampled or compressed.
        outputs: Any
            Your model output.
        losses: List[np.ndarray]
            The loss per point, per streamline.
        mean_per_line: np.ndarray
            The mean loss per streamline.
        eos_probs: List[np.ndarray]
            The probability per point.
        eos_errors: List[np.ndarray]
            The absolute difference between EOS probability and real EOS value.
        mean_per_line_eos: np.ndarray
            The mean eos error per line.
        """
        sft = resample_or_compress(sft, self.model.step_size,
                                   self.model.nb_points,
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
        eos_probs = []
        mean_per_line = None
        eos_errors, mean_per_line_eos = (None, None)

        # Run all batches
        batch_start = 0
        batch_end = batch_size
        with torch.no_grad():
            for _ in tqdm(range(nb_batches), desc="Batches", total=nb_batches):
                # Seems to help.... We need to discover why
                # log_gpu_memory_usage()
                torch.cuda.empty_cache()

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
                outputs = self.model.merge_batches_outputs(outputs, batch_out,
                                                           device='cpu')

                # 3. Compute loss: not averaged = one tensor of losses per
                # streamline.
                if compute_loss:
                    tmp_losses, n, tmp_eos_probs = self.model.compute_loss(
                        batch_out, streamlines, average_results=False,
                        return_eos_probs=True)
                    losses.extend([loss.cpu().numpy() for loss in tmp_losses])
                    if tmp_eos_probs is not None:
                        eos_probs.extend([eos_loss.cpu().numpy()
                                          for eos_loss in tmp_eos_probs])

                # Prepare next batch
                batch_start = batch_end
                batch_end = min(batch_start + batch_size, len(sft))

        if compute_loss:
            mean_per_line = np.asarray([np.mean(loss) for loss in losses])
            # \u00B1 is the plus or minus sign.
            print("Losses: ")
            print(u"    Average per streamline for the {} streamlines is: "
                  u"{:.4f} \u00B1 {:.4f}. Range: [{:.4f} - {:.4f}]"
                  .format(len(sft), np.mean(mean_per_line),
                          np.std(mean_per_line), np.min(mean_per_line),
                          np.max(mean_per_line)))

            tmp = np.hstack(losses)
            print(u"    Mean, averaged over all {} points, is: "
                  u"{:.4f} \u00B1 {:.4f}. Range: [{:.4f} - {:.4f}]"
                  .format(len(tmp), np.mean(tmp), np.std(tmp),
                          np.min(tmp), np.max(tmp)))

            if self.model.direction_getter.add_eos:
                # Expecting prob 0 everywhere and prob 1 at last position
                eos_errors = [
                    np.abs(s - np.asarray([0] * (len(s) - 1) + [1]))
                    for s in eos_probs]

                mean_per_line_eos = np.asarray([np.mean(loss)
                                                for loss in eos_errors])

                print("-------------\n"
                      "EOS error:")
                print(u"    Average EOS error per streamline for the {} "
                      u"streamlines is: {:.4f} \u00B1 {:.4f}. "
                      u"Range: [{:.4f} - {:.4f}]"
                      .format(len(sft), np.mean(mean_per_line_eos),
                              np.std(mean_per_line_eos),
                              np.min(mean_per_line_eos),
                              np.max(mean_per_line_eos)))

                tmp = np.hstack(eos_errors)
                print(u"    Mean, EOS error averaged over all {} points, is: "
                      u"{:.4f} \u00B1 {:.4f}. Range: [{:.4f} - {:.4f}]"
                      .format(len(tmp), np.mean(tmp), np.std(tmp),
                              np.min(tmp), np.max(tmp)))

        return (sft, outputs, losses, mean_per_line,
                eos_probs, eos_errors, mean_per_line_eos)

    def _prepare_inputs(self, streamlines):
        return None


class TesterOneInput(TesterWithDirectionGetter):
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
