# -*- coding: utf-8 -*-
import datetime
import json
import os
import time

import numpy as np
import torch

from dwi_ml.data.dataset.multi_subject_containers import (MultiSubjectDataset)
from dwi_ml.experiment.timer import Timer
from dwi_ml.experiment.monitoring import ValueHistoryMonitor


class DWIMLTrainerAbstractLocal:
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


class DWIMLTrainerAbstractSequences:
    """ Meant for projects working on learning tractography. Information will
    be X = sequences."""
    def __init__(self, training_dataset: MultiSubjectDataset,
                 validation_dataset: MultiSubjectDataset, name: str,
                 hdf5_filename: str, max_epochs: int,
                 patience: int, batch_size: int, volumes_per_batch: int,
                 cycles_per_volume: int, device: torch.device,
                 num_cpu_workers: int, taskman_managed: bool, seed: int):
        """
        Parameters
        ----------
        training_dataset: MultiSubjectDataset
            Instantiated class used for your training data.
        validation_dataset: MultiSubjectDataset
            Instantiated class used for your validation data. Can be None.
        name: str
            The experiment name,
        hdf5_filename: str
            The hdf5 file from which to load the data
        max_epochs: int
            Maximum number of epochs.
        patience: int
            Use early stopping. Defines the number of epochs after which the
            model should stop if the loss hasn't improved.
        batch_size: int
            Number of streamline points per batch.
        volumes_per_batch: int
            Limits the number of volumes used in a batch. If None, will use
            true random sampling.
        cycles_per_volume:  int
            Relevant only if epochs:batch:volumes_used is not null. Number of
            cycles before changing to new volumes. (None is equivalent to 1).
        num_cpu_workers: int
            Number of parallel CPU workers.
        taskman_managed: bool
            If True, taskman manages the experiment. Do not output progress
            bars and instead output special messages for taskman.
        seed: int
            Random experiment seed.
        """

        # Experiment
        self.train_dataset = training_dataset
        self.valid_dataset = validation_dataset
        self.name = name
        self.hdf5_path = hdf5_filename

        # Training epochs:
        self.max_epochs = max_epochs
        self.patience = patience

        # Training batchs:
        self.batch_size = batch_size
        self.volumes_per_batch = volumes_per_batch
        self.cycles_per_volume_batch = cycles_per_volume

        # Memory:
        self.device = device
        self.num_cpu_workers = num_cpu_workers
        self.taskman_managed = taskman_managed
        self.taskman_report = {
            'loss_train': None,
            'loss_valid': None,
            'epoch': None,
            'best_epoch': None,
            'best_score': None,
            'update': None,
            'update_loss': None
        }

        # Time limited run
        self.hangup_time = None
        htime = os.environ.get('HANGUP_TIME', None)
        if htime is not None:
            self.hangup_time = int(htime)
            print('Will hang up at ' + htime)

        # Set random numbers
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        torch.manual_seed(self.seed)  # Set torch seed
        if self.use_gpu:
            torch.cuda.manual_seed(self.seed) # Says error, but ok.

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
            self.train_dataset.load_data()

            input_size = self._compute_input_size()

            self.input_size = input_size
            self.sh_order = self.train_dataset.sh_order

        with Timer("Loading validation dataset", newline=True, color='blue'):
            self.valid_dataset.load_data()

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
