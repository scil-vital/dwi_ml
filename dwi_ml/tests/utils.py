import logging
import os

import torch
from scilpy.io.fetcher import fetch_data, get_home

from dwi_ml.models.main_models import MainModelAbstract, MainModelWithPD, \
    MainModelForTracking
from dwi_ml.tests.expected_values import (
    TEST_EXPECTED_STREAMLINE_GROUPS, TEST_EXPECTED_VOLUME_GROUPS)
from dwi_ml.training.batch_samplers import DWIMLBatchIDSampler
from dwi_ml.training.batch_loaders import DWIMLBatchLoaderOneInput


def fetch_testing_data():
    # Note. Data is accessible because its permission is "anyone with the link"
    # but it can happen that github had trouble accessing it, for unclear
    # reasons. Pushing a new commit (restarting the test) had a good chance to
    # help.
    name_as_dict = {
        'data_for_tests_dwi_ml.zip':
            ['1beRWAorhaINCncttgwqVAP2rNOfx842Q',
             '593f0a7dd5bc0007360eb971e456ccbc']}
    fetch_data(name_as_dict)
    home = get_home()
    testing_data_dir = os.path.join(home, 'data_for_tests_dwi_ml')

    return testing_data_dir


class ModelForTest(MainModelAbstract):
    def __init__(self, experiment_name: str = 'test',
                 normalize_directions: bool = True,
                 neighborhood_type: str = None, neighborhood_radius=None,
                 log_level=logging.root.level):
        super().__init__(experiment_name, normalize_directions,
                         neighborhood_type, neighborhood_radius, log_level)
        self.fake_parameter = torch.nn.Parameter(torch.tensor(42.0))

    def compute_loss(self, model_outputs, streamlines):
        return self.fake_parameter

    def get_tracking_direction_det(self, regressed_dirs: torch.Tensor):
        return regressed_dirs

    def sample_tracking_direction_prob(self, regressed_dir):
        raise NotImplementedError("(Fake) Regression does not allow prob "
                                  "tracking.")

    def forward(self, x: list):
        _ = self.fake_parameter
        regressed_dir = [1., 1., 1.]

        return [regressed_dir for _ in x]


class TrackingModelForTestWithPD(MainModelWithPD, MainModelForTracking):
    def __init__(self, experiment_name: str = 'test',
                 nb_previous_dirs: int = 1,
                 prev_dirs_embedding_size: int = None,
                 prev_dirs_embedding_key: str = 'no_embedding',
                 normalize_targets: bool = True,
                 neighborhood_type: str = None, neighborhood_radius=None,
                 log_level=logging.root.level):
        super().__init__(experiment_name=experiment_name,
                         nb_previous_dirs=nb_previous_dirs,
                         prev_dirs_embedding_size=prev_dirs_embedding_size,
                         prev_dirs_embedding_key=prev_dirs_embedding_key,
                         normalize_targets=normalize_targets,
                         neighborhood_type=neighborhood_type,
                         neighborhood_radius=neighborhood_radius,
                         log_level=log_level)
        self.fake_parameter = torch.nn.Parameter(torch.tensor(42.0))

    def compute_loss(self, model_outputs, streamlines):
        return self.fake_parameter

    def get_tracking_direction_det(self, regressed_dirs,
                                   streamline_lengths=None):
        return regressed_dirs

    def sample_tracking_direction_prob(self, regressed_dir,
                                       streamline_lengths=None):
        raise NotImplementedError("(Fake) Regression does not allow prob "
                                  "tracking.")

    def forward(self, x, streamlines):
        dirs = self.format_directions(streamlines)
        _ = self.compute_and_embed_previous_dirs(dirs)
        _ = self.fake_parameter
        regressed_dir = [1., 1., 1.]

        return [regressed_dir for _ in x]


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
        subset, step_size=None, compress=False, noise_size=0.,
        noise_variability=0., split_ratio=0., reverse_ratio=0.,
        wait_for_gpu=True, log_level=logging.DEBUG):

    logging.debug('    Initializing batch loader...')
    batch_loader = DWIMLBatchLoaderOneInput(
        subset, TEST_EXPECTED_VOLUME_GROUPS[0],
        TEST_EXPECTED_STREAMLINE_GROUPS[0], rng=1234,
        compress=compress, step_size=step_size, split_ratio=split_ratio,
        noise_gaussian_size_training=noise_size,
        noise_gaussian_var_training=noise_variability,
        noise_gaussian_size_validation=0,
        noise_gaussian_var_validation=0,
        reverse_ratio=reverse_ratio,
        neighborhood_points=None, wait_for_gpu=wait_for_gpu,
        log_level=log_level)

    return batch_loader
