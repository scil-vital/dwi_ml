import logging

from dwi_ml.models.main_models import MainModelAbstract
from dwi_ml.tests.expected_values import (
    TEST_EXPECTED_STREAMLINE_GROUPS, TEST_EXPECTED_VOLUME_GROUPS)
from dwi_ml.training.batch_samplers import DWIMLBatchSampler
from dwi_ml.training.batch_loaders import BatchLoaderOneInput


class ModelForTest(MainModelAbstract):
    def __init__(self):
        super().__init__('test', normalize_directions=True,
                         neighborhood_type=None, neighborhood_radius=None)

    def compute_loss(self, model_outputs, streamlines, device):
        return 0.3

    def get_tracking_direction_det(self, model_outputs):
        return [[1., 1., 1.]]

    def sample_tracking_direction_prob(self, model_outputs):
        return [[1., 1., 1.]]


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
    batch_sampler = DWIMLBatchSampler(
        subset, TEST_EXPECTED_STREAMLINE_GROUPS[0],
        batch_size=batch_size, batch_size_units=batch_size_units,
        nb_streamlines_per_chunk=chunk_size,
        rng=test_default_rng,
        nb_subjects_per_batch=test_default_nb_subjects_per_batch,
        cycles=test_default_cycles, log_level=log_level)

    return batch_sampler


def create_batch_loader(
            subset, step_size=None, compress=False, noise_size=0.,
            noise_variability=0.,
            split_ratio=0., reverse_ratio=0., wait_for_gpu=True):

    logging.debug('    Initializing batch loader...')
    batch_loader = BatchLoaderOneInput(
        subset, TEST_EXPECTED_VOLUME_GROUPS[0],
        TEST_EXPECTED_STREAMLINE_GROUPS[0], rng=1234,
        compress=compress, step_size=step_size, split_ratio=split_ratio,
        noise_gaussian_size=noise_size,
        noise_gaussian_variability=noise_variability,
        reverse_ratio=reverse_ratio,
        neighborhood_points=None, wait_for_gpu=wait_for_gpu,
        log_level=logging.INFO)

    return batch_loader
