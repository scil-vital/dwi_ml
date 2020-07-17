# -*- coding: utf-8 -*-
import datetime
import json
import os
import time

import numpy as np
import torch

from dwi_ml.data.dataset.data_list import (DataListForTorch,
                                           LazyDataListForTorch)
from dwi_ml.experiment.timer import Timer
from dwi_ml.experiment.monitoring import ValueHistoryMonitor


class DWIMLAbstractLocal:
    """ Meant for projects working on learning local information in the
    voxel. Information will be X = a voxel. """
    def __init__(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def train(self, **kwargs):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load_model(self, filepath, **kwargs):
        raise NotImplementedError


class DWIMLAbstractSequences:
    """ Meant for projects working on learning tractography. Information will
    be X = sequences."""
    def __init__(self, args):
        """
        args: dictionary with a list of parameters. See the yaml file for a
        better description of parameters. Should contain:
            - Experiment: 'name',
            - Dataset: 'hdf5_filename', 'training_subjs_filename',
                'validation_subjs_filename'
            - Preprocessing: 'step_size'
            - Data augmentation: 'add_noise', 'split_ratio'
            - Input: 'num_previous_dirs' and
                'neighborhood_type': str, the choice of neighborhood between
                    'sphere' or 'grid'.
                'neighborhood_radius': int or float, the neighborhood radius.
            - Training epochs: 'max_epochs', 'patience'
            - Training batchs: 'batch_size', 'volumes_per_batch',
                'cycles_per_volume'
            - Memory: 'lazy', 'cache_manager', 'use_gpu', 'num_cpu_workers',
                'worker_interpolation', 'taskman_managed'
            - Randomization: 'seed'
        """

        # Experiment
        self.name = args['name']

        # Dataset
        self.hdf5_path = args['hdf5_filename']
        self.training_subjs = self._read_subjects(
            args['training_subjs_filename'])
        self.validation_subjs = self._read_subjects(
            args['validation_subs_filename'])

        # Preprocessing:
        self.step_size = args['step_size']

        # Data augmentation:
        self.add_noise = args['add_noise']
        self.split_ratio = args['split_ratio']

        # Input:
        self.neighborhood_type = args['neighborhood_type']
        self.neighborhood_radius = args['neighborhood_radius']
        self.num_previous_dirs = args['num_previous_dirs']

        # Training epochs:
        self.max_epochs = args['max_epochs']
        self.patience = args['patience']

        # Training batchs:
        self.batch_size = args['batch_size']
        self.volumes_per_batch = args['volumes_per_batch']
        self.cycles_per_volume_batch = args['cycles_per_volume']

        # Memory:
        self.lazy = args['lazy']
        self.cache_manager = args['cache_manager']
        self.use_gpu = args['use_gpu']
        self.num_cpu_workers = args['num_cpu_workers']
        self.worker_interpolation = args['worker_interpolation']
        self.taskman_managed = args['taskman_managed']
        self.taskman_report = {
            'loss_train': None,
            'loss_valid': None,
            'epoch': None,
            'best_epoch': None,
            'best_score': None,
            'update': None,
            'update_loss': None
        }

        # Randomization:
        self.seed = args['seed']

        # Time limited run
        self.hangup_time = None
        htime = os.environ.get('HANGUP_TIME', None)
        if htime is not None:
            self.hangup_time = int(htime)
            print('Will hang up at ' + htime)

        # Set device
        self.device = None
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Set random numbers
        self.rng = np.random.RandomState(self.seed)
        torch.manual_seed(self.seed)  # Set torch seed
        if self.use_gpu:
            torch.cuda.manual_seed(self.seed) # Says error, but ok.

        # If using worker_interpolation, data is processed on CPU
        self.dataset_device = torch.device(
            'cpu') if self.worker_interpolation else self.device

        # Init datasets
        # NOTE. WE HOPE THAT MULTISUBJECT CAN REALLY BE COMMON TO ALL OF US.
        # So, I've pu the dataset creation here in the abstract. Else we can
        # bring it back to each user's script.
        other_kw_args = {}
        if self.lazy:
            dataset_cls = LazyMultiSubjectDataset
            if self.cache_manager:
                other_kw_args['cache_size'] = self.volumes_per_batch
        else:
            dataset_cls = MultiSubjectDataset

        self.train_dataset = dataset_cls(
            self.train_database_path, self.rng,
            add_streamline_noise=self.add_streamline_noise,
            step_size=self.step_size,
            neighborhood_dist_mm=self.neighborhood_dist_mm,
            streamlines_cut_ratio=self.streamlines_cut_ratio,
            add_previous_dir=self.add_previous_dir,
            do_interpolation=self.worker_interpolation,
            device=self.dataset_device,
            taskman_managed=self.taskman_managed,
            **other_kw_args)
        self.valid_dataset = dataset_cls(
            self.valid_database_path, self.rng,
            add_streamline_noise=False,
            step_size=self.step_size,
            neighborhood_dist_mm=self.neighborhood_dist_mm,
            streamlines_cut_ratio=None,
            add_previous_dir=self.add_previous_dir,
            do_interpolation=self.worker_interpolation,
            device=self.dataset_device,
            taskman_managed=self.taskman_managed,
            **other_kw_args)

        # Other variables
        self.sh_order = None  # Will be set once the dataset is loaded
        self.input_size = None  # Will be set once the dataset is loaded

        self.current_epoch = 0

        self.experiment_dir = (self.name if self.name
                               else datetime.datetime.now().strftime(
            "%Y_%m_%d_%H%M%S")) + '_' + type(self).__name__

        self.optimizer = None   # Will be defined later with ADAM
        self.model = None       # Will be defined by the main user

        # Setup monitors
        self.train_loss_monitor = ValueHistoryMonitor("Training loss")
        self.valid_loss_monitor = ValueHistoryMonitor("Validation loss")

    @staticmethod
    def _read_subjects(subjects_filename):
        #toDo
        raise NotImplementedError

    def train(self, **kwargs):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load_model(self, filepath, **kwargs):
        raise NotImplementedError

    def load_dataset(self):
        """
        This method loads the data (streamlines and data volume).
        """
        with Timer("Loading training dataset", newline=True, color='blue'):
            self.train_dataset.load()

            input_size = self._compute_input_size()

            self.input_size = input_size
            self.sh_order = self.train_dataset.sh_order

        with Timer("Loading validation dataset", newline=True, color='blue'):
            self.valid_dataset.load()

    def _compute_input_size(self):
        # Basic input size
        expected_input_size = self.train_dataset.multisubject_manager.feature_size

        # + neighbors
        if self.neighborhood_dist_mm:
            expected_input_size += \
                self.nb_neighborhood_axes * \
                self.train_dataset.multisubject_manager.feature_size

        # + previous direction
        if self.add_previous_dir:
            expected_input_size += self.nb_degree_angles

        return expected_input_size

    def _should_quit(self, iter_timer):
        # If:
        #   hang up signal received
        #   time remaining is less than one epoch + 30 seconds
        # exit training.
        return (self.hangup_time is not None and
               time.time() + iter_timer.mean * 2.0 + 30 > self.hangup_time)

    def _update_taskman_report(self, updates):
        self.taskman_report.update(updates)
        self.taskman_report['time'] = time.time()
        print('!taskman' + json.dumps(self.taskman_report), flush=True)
