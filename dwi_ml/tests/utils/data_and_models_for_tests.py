import logging
import os
from typing import List

import torch
from torch import Tensor
from torch.nn.utils.rnn import pack_sequence

from scilpy.io.fetcher import fetch_data, get_home

from dwi_ml.data.processing.streamlines.post_processing import \
    normalize_directions, compute_directions
from dwi_ml.models.main_models import (
    ModelForTracking, ModelWithNeighborhood, MainModelOneInput,
    ModelWithPreviousDirections)
from dwi_ml.tests.utils.expected_values import (
    TEST_EXPECTED_STREAMLINE_GROUPS, TEST_EXPECTED_VOLUME_GROUPS)
from dwi_ml.tracking.propagator import DWIMLPropagatorwithStreamlineMemory, \
    DWIMLPropagatorOneInput
from dwi_ml.training.batch_samplers import DWIMLBatchIDSampler
from dwi_ml.training.batch_loaders import DWIMLBatchLoaderOneInput


def fetch_testing_data():
    # Note. Data is accessible because its permission is "anyone with the link"
    # but it can happen that github had trouble accessing it, for unclear
    # reasons. Pushing a new commit (restarting the test) had a good chance to
    # help.
    home = get_home()
    testing_data_dir = os.path.join(home, 'data_for_tests_dwi_ml')

    if not os.path.exists(testing_data_dir):
        # Access to the file dwi_ml.zip:
        # https://drive.google.com/uc?id=1beRWAorhaINCncttgwqVAP2rNOfx842Q
        name_as_dict = {
            'data_for_tests_dwi_ml.zip':
                ['1beRWAorhaINCncttgwqVAP2rNOfx842Q',
                 '8bdbf051877ec5c70aace21c9dab9bb7']}
        fetch_data(name_as_dict)

    return testing_data_dir


def create_test_batch():
    logging.debug("Creating batch: 2 streamlines, the first has 4 points "
                  "and the second, 3. Input: 4 features per point.")

    # dwi1 : data for the 3 first points
    flattened_dwi1 = torch.as_tensor([[10., 11., 12., 13.],
                                      [50., 51., 52., 53.],
                                      [60., 62., 62., 63.]])
    streamline1 = torch.as_tensor([[0.1, 0.2, 0.3],
                                   [1.1, 11.2, 1.3],
                                   [2.1, 2.2, 2.3],
                                   [3.1, 3.2, 3.3]])

    # dwi2 : data for the 2 first points
    flattened_dwi2 = torch.as_tensor([[10., 11., 12., 13.],
                                      [50., 51., 52., 53.]])
    streamline2 = torch.as_tensor([[10.1, 10.2, 10.3],
                                   [11.1, 11.2, 11.3],
                                   [12.1, 12.2, 12.3]])

    batch_x_training = [flattened_dwi1, flattened_dwi2]
    batch_s_training = [streamline1, streamline2]

    batch_x_tracking = [flattened_dwi1[0:2, :], flattened_dwi2[0:2, :]]
    batch_s_tracking = [streamline1[0:2, :], streamline2[0:2, :]]

    return (batch_x_training, batch_x_tracking,
            batch_s_training, batch_s_tracking)


class ModelForTest(MainModelOneInput, ModelWithNeighborhood):
    def __init__(self, experiment_name: str = 'test',
                 neighborhood_type: str = None, neighborhood_radius=None,
                 log_level=logging.root.level):
        super().__init__(experiment_name=experiment_name,
                         neighborhood_type=neighborhood_type,
                         neighborhood_radius=neighborhood_radius,
                         log_level=log_level)
        self.fake_parameter = torch.nn.Parameter(torch.as_tensor(42.0))

    def compute_loss(self, model_outputs, target_streamlines=None):
        mean = self.fake_parameter
        n = 30
        return mean, n

    def forward(self, x: list):
        _ = self.fake_parameter
        regressed_dir = torch.as_tensor([1., 1., 1.])

        return [regressed_dir for _ in x]


class TrackingModelForTestWithPD(ModelWithPreviousDirections, ModelForTracking,
                                 ModelWithNeighborhood, MainModelOneInput):
    def __init__(self, experiment_name: str = 'test',
                 neighborhood_type: str = None, neighborhood_radius=None,
                 log_level=logging.root.level,
                 # PREVIOUS DIRS
                 nb_previous_dirs=0, prev_dirs_embedding_size=None,
                 prev_dirs_embedding_key=None, normalize_prev_dirs=True,
                 # DIRECTION GETTER
                 dg_key='cosine-regression', dg_args=None,
                 dg_input_size=4, normalize_targets=True,
                 normalize_outputs=True):

        super().__init__(
            experiment_name=experiment_name,
            neighborhood_type=neighborhood_type,
            neighborhood_radius=neighborhood_radius, log_level=log_level,
            # For super MainModelWithPD:
            nb_previous_dirs=nb_previous_dirs,
            prev_dirs_embedding_size=prev_dirs_embedding_size,
            prev_dirs_embedding_key=prev_dirs_embedding_key,
            normalize_prev_dirs=normalize_prev_dirs,
            # For super MainModelForTracking:
            normalize_targets=normalize_targets, dg_key=dg_key,
            dg_args=dg_args)

        # If multiple inheritance goes well, these params should be set
        # correctly
        assert self.model_uses_streamlines

        self.instantiate_direction_getter(dg_input_size)

    def compute_loss(self, model_outputs: torch.tensor,
                     target_streamlines: List[torch.Tensor], **kw):
        target_dirs = compute_directions(target_streamlines)

        if self.normalize_targets:
            target_dirs = normalize_directions(target_dirs)

        # Packing dirs and using the .data instead of looping on streamlines.
        # Anyway, loss is computed point by point.
        target_dirs = torch.cat(target_dirs)

        # Computing loss
        # Depends on model. Ex: regression: direct difference.
        # Classification: log-likelihood.
        # Gaussian: difference between distribution and target.
        return self.direction_getter.compute_loss(model_outputs, target_dirs)

    def get_tracking_directions(self, regressed_dirs, algo):
        if algo == 'det':
            return regressed_dirs.detach()
        elif algo == 'prob':
            raise NotImplementedError(
                "Our test model uses (fake) regression and does not allow "
                "prob tracking.")
        else:
            raise ValueError("'algo' should be 'det' or 'prob'.")

    def forward(self, inputs: List[torch.tensor],
                target_streamlines: List[torch.tensor],
                hidden_reccurent_states: tuple = None,
                return_state: bool = False, is_tracking: bool = False,
                ) -> List[torch.tensor]:
        target_dirs = compute_directions(target_streamlines)

        assert len(target_dirs) == len(inputs), \
            ("Error. The target directions contain {} streamlines but the "
             "input contains {}").format(len(target_dirs), len(inputs))

        # Previous dirs
        n_prev_dirs_embedded = self.normalize_and_embed_previous_dirs(
            target_dirs)
        if n_prev_dirs_embedded is not None:
            assert len(n_prev_dirs_embedded) == len(target_dirs)

        # Fake intermediate layer
        model_outputs = [torch.ones(len(s), self.direction_getter.input_size,
                                    device=self.device)
                         for s in inputs]

        # Packing results
        # Resulting shape = self.df_input_size *
        model_outputs = pack_sequence(model_outputs, enforce_sorted=False).data

        # Direction getter
        model_outputs = self.direction_getter(model_outputs)
        if self.normalize_outputs:
            model_outputs = normalize_directions([model_outputs])

        return model_outputs


def create_test_batch_sampler(
        subset, batch_size, batch_size_units, chunk_size=None,
        log_level=logging.DEBUG):
    """
    Create a batch sampler and a batch loader for testing.
    """
    test_default_nb_subjects_per_batch = 1
    test_default_cycles = 1
    test_default_rng = 1234

    logging.debug('    Initializing batch sampler...')
    batch_sampler = DWIMLBatchIDSampler(
        subset, TEST_EXPECTED_STREAMLINE_GROUPS[0],
        batch_size_training=batch_size, batch_size_validation=0,
        batch_size_units=batch_size_units,
        nb_streamlines_per_chunk=chunk_size,
        rng=test_default_rng,
        nb_subjects_per_batch=test_default_nb_subjects_per_batch,
        cycles=test_default_cycles, log_level=log_level)

    return batch_sampler


def create_batch_loader(
        subset, model, step_size=None, compress=False, noise_size=0.,
        noise_variability=0., split_ratio=0., reverse_ratio=0.,
        wait_for_gpu=True, log_level=logging.DEBUG):
    logging.debug('    Initializing batch loader...')
    batch_loader = DWIMLBatchLoaderOneInput(
        dataset=subset, input_group_name=TEST_EXPECTED_VOLUME_GROUPS[0],
        streamline_group_name=TEST_EXPECTED_STREAMLINE_GROUPS[0], rng=1234,
        compress=compress, step_size=step_size, split_ratio=split_ratio,
        noise_gaussian_size_training=noise_size,
        noise_gaussian_var_training=noise_variability,
        noise_gaussian_size_validation=0,
        noise_gaussian_var_validation=0,
        reverse_ratio=reverse_ratio, wait_for_gpu=wait_for_gpu,
        log_level=log_level, model=model)

    return batch_loader


class TestPropagator(DWIMLPropagatorwithStreamlineMemory,
                     DWIMLPropagatorOneInput):
    def __init__(self, **kw):
        super().__init__(**kw)
