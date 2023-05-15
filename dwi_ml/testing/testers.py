# -*- coding: utf-8 -*-
import json
import logging
import os

import numpy as np
import torch

from dwi_ml.data.processing.streamlines.data_augmentation import \
    resample_or_compress
from dwi_ml.models.main_models import ModelWithDirectionGetter, MainModelOneInput
from dwi_ml.testing.utils import prepare_dataset_one_subj

logger = logging.getLogger('tester_logger')


class Tester:
    """
    Similar to the trainer, it loads the data, runs the model and gets the
    loss.

    However, the reference streamlines are loaded from a given SFT rather than
    from the hdf5. This choice allows to test the loss on various bundles for
    a better interpretation of the models' performances.
    """
    def __init__(self, experiment_path: str, model: ModelWithDirectionGetter,
                 batch_size: int = None, device: torch.device = None):
        """
        Parameters
        ----------
        experiment_path: str
        model: ModelWithDirectionGetter

        batch_size: int
        device: torch.Device
        """
        self.experiment_path = experiment_path
        self.device = device
        self.model = model
        self.model.set_context('visu')
        self.model.eval()  # Removes dropout.
        self.model.move_to(device)
        self.batch_size = batch_size

        self.subset = None
        self.subj_idx = None
        self.experiment_params = None
        self.input_info = None
        self._params = None

    @property
    def params(self):
        if self._params is None:
            logging.info("Loading information from checkpoint")
            params_filename = os.path.join(self.experiment_path, "parameters.json")
            with open(params_filename, 'r') as json_file:
                self._params = json.load(json_file)
        return self._params

    @property
    def streamlines_group(self):
        return self.params["Loader params"]["streamline_group_name"]

    def load_and_format_data(self, subj_id, hdf5_file, subset_name):
        """
        Loads the data associated with subj_id in the hdf5 file and formats it
        as stated by the experiment's batch loader's params.

        Returns: SFT, all streamlines for this subject.
        """
        # 1. Load subject
        logging.info("Loading subject {} from hdf5.".format(subj_id))
        self.subset, self.subj_idx = prepare_dataset_one_subj(
            hdf5_file, subj_id, subset_name=subset_name,
            volume_groups=self._volume_groups,
            streamline_groups=[self.streamlines_group],
            lazy=False, cache_size=1)

        # 2. Load SFT as in dataloader, except we don't loop on many subject,
        # we don't verify streamline ids (loading all), and we don't split /
        # reverse streamlines. But we resample / compress.
        logging.info("Loading its streamlines as SFT.")
        streamline_group_idx = self.subset.streamline_groups.index(
            self.streamlines_group)
        subj_data = self.subset.subjs_data_list.get_subj_with_handle(self.subj_idx)
        subj_sft_data = subj_data.sft_data_list[streamline_group_idx]
        sft = subj_sft_data.as_sft()

        sft = resample_or_compress(sft, self.model.step_size,
                                   self.model.compress_lines)
        sft.to_vox()
        sft.to_corner()

        return sft

    @property
    def _volume_groups(self):
        raise NotImplementedError

    def run_model_on_sft(self, sft, add_zeros_if_no_eos=True,
                         compute_loss=True, uncompress_loss=False,
                         force_compress_loss=False):
        """
        Equivalent of one validation pass.

        Parameters
        ----------
        sft: StatefulTractogram
        add_zeros_if_no_eos: bool
            If true, the loss with no zeros is also printed.
        compute_loss: bool
            If False, the method returns the outputs after a forward pass only.
        uncompress_loss: bool
            If true, ignores the --compress_loss option of the direction
            getter.
        force_compress_loss: bool
            If true, compresses the loss even if that is not the model's
            parameter.
        """
        if uncompress_loss and force_compress_loss:
            raise ValueError("Can't use both compress and uncompress.")

        save_val = self.model.direction_getter.compress_loss
        if uncompress_loss:
            self.model.direction_getter.compress_loss = False
        elif force_compress_loss:
            self.model.direction_getter.compress_loss = True
            self.model.direction_getter.compress_eps = force_compress_loss

        batch_size = self.batch_size or len(sft)
        nb_batches = int(np.ceil(len(sft) / batch_size))

        losses = []
        compressed_n = []
        outputs = []
        batch_start = 0
        batch_end = batch_size
        with torch.no_grad():
            for batch in range(nb_batches):
                logging.info("  Batch #{}:  {} - {}"
                             .format(batch + 1, batch_start, batch_end))
                # 1. Prepare batch
                streamlines = [
                    torch.as_tensor(s, dtype=torch.float32, device=self.device)
                    for s in sft.streamlines[batch_start:batch_end]]

                if not self.model.direction_getter.add_eos:
                    # We don't use the last coord because it does not have an
                    # associated target direction.
                    streamlines_f = [s[:-1, :] for s in streamlines]
                else:
                    streamlines_f = streamlines
                inputs = self._prepare_inputs_at_pos(streamlines_f)

                # 2. Run forward
                tmp_outputs = self.model(inputs, streamlines_f)
                del streamlines_f

                if compute_loss:
                    # 3. Compute loss
                    tmp_losses = self.model.compute_loss(
                        tmp_outputs, streamlines, average_results=False)

                outputs.extend([o.cpu() for o in tmp_outputs])
                if self.model.direction_getter.compress_loss:
                    tmp_losses, n = tmp_losses
                    losses.append(tmp_losses)
                    compressed_n.append(n)
                else:
                    losses.extend([line_loss.cpu() for line_loss in tmp_losses])

                batch_start = batch_end
                batch_end = min(batch_start + batch_size, len(sft))

            if self.model.direction_getter.compress_loss:
                total_n = sum(compressed_n)
                total_loss = sum([loss * n for loss, n in
                                  zip(losses, compressed_n)]) / total_n
            else:
                total_n = sum([len(line_loss) for line_loss in losses])
                total_loss = torch.mean(torch.hstack(losses))

            print("Loss function, averaged over all {} points in the chosen "
                  "SFT, is: {}.".format(total_n, total_loss))

            if (not self.model.direction_getter.compress_loss) and \
                    add_zeros_if_no_eos and \
                    not self.model.direction_getter.add_eos:
                zero = torch.zeros(1)
                losses = [torch.hstack([line, zero]) for line in losses]
                total_n = sum([len(line_loss) for line_loss in losses])
                total_loss = torch.mean(torch.hstack(losses))
                print("When adding a 0 loss at the EOS position, the mean "
                      "loss for {} points is {}.".format(total_n, total_loss))

        self.model.direction_getter.compress_loss = save_val

        return outputs, losses

    def _prepare_inputs_at_pos(self, streamlines):
        raise NotImplementedError


class TesterOneInput(Tester):
    model: MainModelOneInput

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.input_group_idx = None

    @property
    def _volume_groups(self):
        return [self.params["Loader params"]["input_group_name"]]

    def load_and_format_data(self, subj_id, hdf5_file, subset_name):
        sft = super().load_and_format_data(subj_id, hdf5_file, subset_name)
        self.input_group_idx = self.subset.volume_groups.index(
            self._volume_groups[0])
        return sft

    def _prepare_inputs_at_pos(self, streamlines):
        return self.model.prepare_batch_one_input(
            streamlines, self.subset, self.subj_idx, self.input_group_idx)
