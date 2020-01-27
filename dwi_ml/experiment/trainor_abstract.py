#!/usr/bin/env python

import datetime
import json
import os

import time

import numpy as np
import torch

from VITALabAI import VITALabAiAbstract
from scil_vital.shared.code.data.multisubject import \
    (MultiSubjectDataset, LazyMultiSubjectDataset)
from scil_vital.shared.code.utils.timer import Timer
from scil_vital.shared.code.utils.experiment_utils import ValueMonitor


class TractoMLAbstractLocal(VITALabAiAbstract):
    def __init__(self):
        super().__init__(dataset=None)
        raise NotImplementedError

    def get_backend(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def train(self, **kwargs):
        raise NotImplementedError

    def predict(self, **kwargs):
        raise NotImplementedError

    def predict_on_batch(self, batch):
        raise NotImplementedError

    def evaluate(self, **kwargs):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load_model(self, filepath, **kwargs):
        raise NotImplementedError

    def load_weights(self, filepath, **kwargs):
        raise NotImplementedError


class TractoMLAbstractSequences(VITALabAiAbstract):
    def __init__(self,
                 train_database_path: str = None,
                 valid_database_path: str = None,
                 name: str = None,
                 # Concerning the choice of inputs:
                 nb_degree_angles: int = 128,
                 add_streamline_noise: bool = False,
                 streamlines_cut_ratio: float = None, step_size: float = None,
                 neighborhood_dist_mm: float = None,
                 nb_neighborhood_axes: int = 6,
                 add_previous_dir: bool = False,
                 lazy: bool = False,
                 # Concerning the memory usage:
                 batch_size: int = 20000, volumes_per_batch: int = None,
                 cycles_per_volume_batch: int = 1,
                 n_epoch: int = 100, seed: int = 1234, patience: int = 20,
                 use_gpu: bool = True, num_workers: int = 0,
                 worker_interpolation: bool = False,
                 cache_manager: bool = False, taskman_managed: bool = False):
        """
        Mandatory parameters:
        ---------------------
        train_database_path : str
            Path to training database (hdf5 file)
        valid_database_path : str
            Path to validation database (hdf5 file)

        Optional parameters:
        --------------------
        ====> General
        name : str
            Optional name of the experiment. If given, it is prepended to the
            auto-generated name. [None]

        ====> Concerning the choice of inputs:
        nb_degree_angles: int
            Precision for angles: nb of directions on the sphere. If previous
            direction is added to input, we need to know how many that is. But
            we manage the output with output_model, not with this option. [128]
        add_streamline_noise : bool
            If set, add random gaussian noise to streamline coordinates
            on-the-fly. Noise variance is 0.1 * step-size, or 0.1mm if no step
            size is used. [False]
        streamlines_cut_ratio : float
            Percentage of streamlines to randomly cut in each batch. If None, do
            not split streamlines. [None]
                NOTE: Preprocessed .hdf5 file should contain resampled
                streamlines; otherwise, cutting streamlines will be biased
                towards long segments (less points)
        step_size : float
            Constant step size that every streamline should have between points
            (in mm). If None, train on streamlines as they are (ex, compressed).
            [None]
        neighborhood_dist_mm : float
            If given, add neighboring information to the input signal at the
            given distance in each axis (in mm). [None]
        neighborhood_axes : int
            Nb of axes at which to get neighborhood distance. Default = 6 (up,
            down, left, right, front, back).
        add_previous_dir : bool
            If set, add the previous streamline direction to the input signal.
            [False]
        lazy : bool
            If True, use a lazy dataset. [False]

        ====> Concerning the memory usage:
        batch_size : int
            Number of time steps to use in a batch (the length of sequences vary
            a lot, so we define the number of time steps to use a more
            consistent amount of memory) [20,000]
        volumes_per_batch : int
            Limit the number of sampled volumes inside a single batch.
            If None, use true random sampling. [None]
        cycles_per_volume_batch : int
            Number of batches where the same volumes will be reused before
            sampling new volumes. [1]
        n_epoch : int
            Maximum number of epochs [100]
        seed : int
            Seed for random numbers [1234]
        patience : int
            Use early stopping. Defines the number of epochs after which
            the model should stop training if the loss hasn't improved. [20]
        use_gpu : bool
            Use the GPU; if False, use CPU. [True]
        num_workers : int
            Number of processes that should process the data between training
            updates. [0]
        worker_interpolation : bool
            If True and num_workers > 0, interpolation will be done on CPU by
            the workers. Otherwise, interpolation is done on the main thread
            using the chosen device. [False]
        cache_manager : bool
            If True, use a cache manager to keep volumes and streamlines in
            memory. [False]
        taskman_managed : bool
            If True, taskman manages the experiment. Do not output progress
            bars and instead output special messages for taskman. [False]
        """

        # Init mandatory properties
        self.train_database_path = train_database_path
        self.valid_database_path = valid_database_path

        # Init optional properties
        self.name = name

        # Init "globals" from user's project
        self.nb_degree_angles = nb_degree_angles

        # Init args concerning choice of inputs
        self.add_streamline_noise = add_streamline_noise
        self.streamlines_cut_ratio = streamlines_cut_ratio
        self.step_size = step_size
        self.neighborhood_dist_mm = neighborhood_dist_mm
        self.nb_neighborhood_axes = nb_neighborhood_axes                                                # toDo. À voir!! Je vais peut-être devoir changer int pour str='method'
                                                                                                        #  On aurait la méthode "6axes" et la méthode "mimicGrid" pour mon CNN
                                                                                                        #  où je prendrais 27 axes, pas tous de la même longueur! Possiblement
                                                                                                        #  le double d'axe pour avoir l'équivalent de 2 voxels autour de mon point
                                                                                                        #  dans toutes les directions. Ça pourrait être [str|int]
        self.add_previous_dir = add_previous_dir
        self.lazy = lazy

        # Init args concerning memory usage
        self.batch_size = int(batch_size)
        self.volumes_per_batch = volumes_per_batch
        self.n_epoch = int(n_epoch)
        self.seed = seed
        self.patience = patience
        self.use_gpu = use_gpu
        self.num_workers = num_workers
        self.worker_interpolation = worker_interpolation
        self.cycles_per_volume_batch = cycles_per_volume_batch
        self.cache_manager = cache_manager
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
            torch.cuda.manual_seed(self.seed)                                                           # toDo. Pourquoi ça dit error?

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
        self.train_loss_monitor = ValueMonitor("Training loss")
        self.valid_loss_monitor = ValueMonitor("Validation loss")
        self.grad_norm_monitor = ValueMonitor("Grad Norm")                                          # ToDo Est-ce que tout le monde utilise grad norm??

    def get_backend(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def train(self, **kwargs):
        raise NotImplementedError
                                                                                                # ToDo: "train" depends on each user, but can we define
                                                                                                #  sub-functions here that could encapsulate some sub-tasks that
                                                                                                #  everybody uses? One day we could compare our codes.

    def predict(self, **kwargs):
        raise NotImplementedError

    def predict_on_batch(self, batch):
        raise NotImplementedError

    def evaluate(self, **kwargs):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load_model(self, filepath, **kwargs):
        raise NotImplementedError

    def load_weights(self, filepath, **kwargs):
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

    def model_summarize(self):
        raise NotImplementedError
