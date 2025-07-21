# -*- coding: utf-8 -*-
from datetime import datetime
import json
import logging
import os
import shutil
from typing import Union, List

from comet_ml import (Experiment as CometExperiment, ExistingExperiment)
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from dwi_ml.experiment_utils.memory import log_gpu_memory_usage
from dwi_ml.experiment_utils.tqdm_logging import tqdm_logging_redirect
from dwi_ml.models.main_models import (MainModelAbstract,
                                       ModelWithDirectionGetter)
from dwi_ml.training.batch_loaders import (
    DWIMLStreamlinesBatchLoader, DWIMLBatchLoaderOneInput)
from dwi_ml.training.batch_samplers import DWIMLBatchIDSampler
from dwi_ml.training.utils.gradient_norm import compute_gradient_norm
from dwi_ml.training.utils.monitoring import (
    BestEpochMonitor, IterTimer, BatchHistoryMonitor, TimeMonitor,
    EarlyStoppingError)

logger = logging.getLogger('train_logger')
# If the remaining time is less than one epoch + X seconds, we will quit
# training now, to allow saving time.
QUIT_TIME_DELAY_SECONDS = 10
# Update comet every 10 batch for faster management.
COMET_UPDATE_FREQUENCY = 10

# toDo. See if we would like to implement mixed precision.
#  Could improve performance / speed
#  https://towardsdatascience.com/optimize-pytorch-performance-for-speed-and-memory-efficiency-2022-84f453916ea6
#  https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/


class DWIMLAbstractTrainer:
    """
    This Trainer class's train_and_validate() method:
        - Creates DataLoaders from the data_loaders. Collate_fn will be the
        found in the batch loader, and the dataset will be found in the data
        sampler.
        - Trains each epoch by using the model's loss computation method.

    Comet is used to save training information, but some logs will also be
    saved locally in the saving_path.

    NOTE: TRAINER USES STREAMLINES COORDINATES IN VOXEL SPACE, CORNER ORIGIN.
    """
    def __init__(self,
                 model: MainModelAbstract, experiments_path: str,
                 experiment_name: str, batch_sampler: DWIMLBatchIDSampler,
                 batch_loader: DWIMLStreamlinesBatchLoader,
                 learning_rates: Union[List, float] = None,
                 weight_decay: float = 0.01,
                 optimizer: str = 'Adam', max_epochs: int = 10,
                 max_batches_per_epoch_training: int = 1000,
                 max_batches_per_epoch_validation: Union[int, None] = 1000,
                 patience: int = None, patience_delta: float = 1e-6,
                 nb_cpu_processes: int = 0, use_gpu: bool = False,
                 clip_grad: float = None,
                 comet_workspace: str = None, comet_project: str = None,
                 from_checkpoint: bool = False, log_level=logging.root.level):
        """
        Parameters
        ----------
        model: MainModelAbstract
            Instantiated class containing your model.
        experiments_path: str
            Path where to save this experiment's results and checkpoints.
            Will be saved in experiments_path/experiment_name.
        experiment_name: str
            Name of this experiment. This will also be the name that will
            appear online for comet.ml experiment.
        batch_sampler: DWIMLBatchIDSampler
            Instantiated class used for sampling batches.
            Data in batch_sampler.dataset must be already loaded.
        batch_loader: DWIMLStreamlinesBatchLoader
            Instantiated class with a load_batch method able to load data
            associated to sampled batch ids. Data in batch_sampler.dataset must
            be already loaded.
        learning_rates: List
            List of at least one learning rate, or None (will use
            torch's default, 0.001). A list [0.01, 0.01, 0.001], for instance,
            would use these values for the first 3 epochs, and keep the final
            value for remaining epochs.
        weight_decay: float
            Add a weight decay penalty on the parameters. Default: 0.01.
            (torch's default).
        optimizer: str
            Torch optimizer choice. Current available options are SGD, Adam or
            RAdam.
        max_epochs: int
            Maximum number of epochs. Default = 10, for no good reason.
        max_batches_per_epoch_training: int
            Maximum number of batches per epoch. Default = 10000, for no good
            reason.
        max_batches_per_epoch_validation: int
            Maximum number of batches per epoch. Default = 10000, for no good
            reason.
        patience: int
            Use early stopping. Defines the number of epochs after which the
            model should stop if the loss hasn't improved. Default: None (i.e.
            no early stopping).
        patience_delta: float
            Limit difference between two validation losses to consider that the
            model improved between the two epochs.
        nb_cpu_processes: int
            Number of parallel CPU workers. Use 0 to avoid parallel threads.
            Default : 0.
        use_gpu: bool
            If true, use GPU device when possible instead of CPU.
            Default = False
        clip_grad : float
            The value to which to clip gradients after the backward pass.
            There is no good value here. Default: 1000.
        comet_workspace: str
            Your comet workspace. See our docs/Getting Started for more
            information on comet and its API key. Default= None (comet.ml will
            not be used).
        comet_project: str
             Send your experiment to a specific comet.ml project. Default: None
             (it will be sent to Uncategorized Experiments).
        from_checkpoint: bool
             If true, we do not create the output dir, as it should already
             exist. Default: False.
        """
        # To developers: do not forget that changes here must be reflected
        # in the save_checkpoint method!

        # ----------------------
        # Values given by the user
        # ----------------------
        self.experiments_path = experiments_path
        self.experiment_name = experiment_name
        self.max_epochs = max_epochs
        self.max_batches_per_epochs_train = max_batches_per_epoch_training
        self.max_batches_per_epochs_valid = max_batches_per_epoch_validation
        self.weight_decay = weight_decay
        self.use_gpu = use_gpu
        self.optimizer_key = optimizer
        self.comet_workspace = comet_workspace
        self.comet_project = comet_project
        self.space = 'vox'
        self.origin = 'corner'
        self.clip_grad = clip_grad

        # Learning rate:
        if learning_rates is None:
            self.learning_rates = [0.001]
        elif isinstance(learning_rates, float):
            # Should be a list but we will accept it.
            self.learning_rates = [learning_rates]
        else:
            self.learning_rates = learning_rates

        # Multiprocessing:
        # Currently, Dataloader multiprocessing is not working very well.
        # system error: to many file handles open.
        # toDo: Verify our datasets if it is our fault.
        #  or update torch version and try again, maybe?
        #  See here: https://github.com/pytorch/pytorch/issues/11201
        #  Amongst suggested solutions: Add this for each worker
        #  torch.multiprocessing.set_sharing_strategy('file_system')
        # Fixing nb processes: 1 and 0 do not have the same effect on the
        # dataloader.
        if nb_cpu_processes == 1:
            nb_cpu_processes = 0
        self.nb_cpu_processes = nb_cpu_processes

        # ----------------------
        # Instantiated classes given by the user
        # ----------------------
        self.batch_sampler = batch_sampler
        self.batch_loader = batch_loader
        self.model = model

        # ----------------------
        # Checks
        # ----------------------
        # Trainer's logging level can be changed separately from main
        # scripts.
        logger.setLevel(log_level)
        self.saving_path = os.path.join(experiments_path, experiment_name)
        self.log_dir = os.path.join(self.saving_path, "logs")
        if not os.path.isdir(experiments_path):
            raise NotADirectoryError(
                "The experiments path does not exist! ({}). Can't create this "
                "experiment sub-folder!".format(experiments_path))
        if not from_checkpoint:
            if os.path.isdir(self.saving_path):
                raise FileExistsError("Current experiment seems to already "
                                      "exist... Use run from checkpoint to "
                                      "continue training.")
            else:
                logger.info('Creating directory {}'.format(self.saving_path))
                os.mkdir(self.saving_path)
                os.mkdir(self.log_dir)

        assert np.all(
            [param == self.batch_loader.dataset.params_for_checkpoint[key] for
             key, param in
             self.batch_sampler.dataset.params_for_checkpoint.items()])
        if self.batch_sampler.dataset.validation_set.nb_subjects == 0:
            self.use_validation = False
            logger.warning(
                "WARNING! There is no validation set. Loss for best epoch "
                "monitoring will be the training loss. \n"
                "Best practice is to have a validation set.")
        else:
            self.use_validation = True
            if max_batches_per_epoch_validation is None:
                self.max_batches_per_epoch_validation = 1000

        if optimizer not in ['SGD', 'Adam', 'RAdam']:
            raise ValueError("Optimizer choice {} not recognized."
                             .format(optimizer))

        # ----------------
        # Create DataLoaders from the BatchSamplers
        # ----------------
        #   * Before usage, context must be set for the batch sampler and the
        #     batch loader, to use appropriate parameters.
        #   * Pin memory if interpolation is done by workers; this means that
        #     dataloader output is on GPU, ready to be fed to the model.
        #     Otherwise, dataloader output is kept on CPU, and the main thread
        #     sends volumes and coords on GPU for interpolation.
        logger.debug("- Instantiating dataloaders...")
        self.train_dataloader = DataLoader(
            dataset=self.batch_sampler.dataset.training_set,
            batch_sampler=self.batch_sampler,
            num_workers=self.nb_cpu_processes,
            collate_fn=self.batch_loader.load_batch_streamlines,
            pin_memory=self.use_gpu)
        self.valid_dataloader = None
        if self.use_validation:
            self.valid_dataloader = DataLoader(
                dataset=self.batch_sampler.dataset.validation_set,
                batch_sampler=self.batch_sampler,
                num_workers=self.nb_cpu_processes,
                collate_fn=self.batch_loader.load_batch_streamlines,
                pin_memory=self.use_gpu)

        # ----------------------
        # Evolving values. They will need to be updated if initialized from
        # checkpoint.
        # ----------------------

        # A. Device and rng value.
        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                torch.cuda.manual_seed(self.batch_sampler.rng)

                logger.info("We will be using GPU!")
            else:
                raise ValueError("You chose GPU (cuda) device but it is not "
                                 "available!")
        else:
            self.device = torch.device('cpu')
        # Send model to device. Reminder, contrary to tensors, model.to
        # overwrites the model. Must be done before launching the optimizer
        # (it needs to use the cuda tensors if using the GPU.)
        self.model.move_to(device=self.device)

        # B. Current epoch
        self.current_epoch = 0

        # C. Nb of batches per epoch.
        # This will be used to estimate time left for nice prints to user.
        # Will be estimated later on.
        self.nb_batches_train = None
        self.nb_batches_valid = None

        # D. Monitors
        # grad_norm = The total norm (sqrt(sum(params**2))) of parameters
        # before gradient clipping, if any.

        # Training: only one monitor.
        self.train_loss_monitor = BatchHistoryMonitor(
            'train_loss_monitor', weighted=True)

        # Validation: As many supervision losses as we want.
        self.valid_local_loss_monitor = BatchHistoryMonitor(
            'valid_local_loss_monitor', weighted=True)
        self.unclipped_grad_norm_monitor = BatchHistoryMonitor(
            'unclipped_grad_norm_monitor', weighted=False)
        self.grad_norm_monitor = BatchHistoryMonitor(
            'grad_norm_monitor', weighted=False)
        self.training_time_monitor = TimeMonitor('training_time_monitor')
        self.validation_time_monitor = TimeMonitor('validation_time_monitor')
        if not patience:
            patience = np.inf
        self.best_epoch_monitor = BestEpochMonitor(
            'best_epoch_monitor', patience, patience_delta)
        self.monitors = [self.train_loss_monitor,
                         self.valid_local_loss_monitor,
                         self.unclipped_grad_norm_monitor,
                         self.grad_norm_monitor, self.training_time_monitor,
                         self.validation_time_monitor, self.best_epoch_monitor]
        self.training_monitors = [self.train_loss_monitor,
                                  self.unclipped_grad_norm_monitor,
                                  self.grad_norm_monitor,
                                  self.training_time_monitor]
        self.validation_monitors = [self.valid_local_loss_monitor,
                                    self.validation_time_monitor]

        # E. Comet Experiment
        # Values will be instantiated in train().
        self.comet_exp = None
        self.comet_key = None

        # ----------------------
        # F. Launching optimizer!
        # ----------------------

        # Build optimizer (Optimizer is built here since it needs the model
        # parameters)
        list_params = [n for n, _ in self.model.named_parameters()]
        logger.debug("Initiating trainer: {}".format(type(self)))
        logger.debug("This trainer will use {} optimization on the "
                     "following model.parameters: {}"
                     .format(self.optimizer_key, list_params))

        if self.optimizer_key == 'RAdam':
            cls = torch.optim.RAdam
        elif self.optimizer_key == 'Adam':
            cls = torch.optim.Adam
        else:
            cls = torch.optim.SGD

        # Learning rate will be set at each epoch.
        self.optimizer = cls(self.model.parameters(),
                             weight_decay=weight_decay)

    @property
    def params_for_checkpoint(self):
        """
        Returns the parameters necessary to initialize an identical Trainer.
        However, the trainer's state could need to be updated (see checkpoint
        management).
        """
        # Not saving experiment_path and experiment_name. Allowing user to
        # move the experiment on his computer between training sessions.

        # Patience is not saved here: we manage it separately to allow the
        # user to increase the patience when running again.
        params = {
            'learning_rates': self.learning_rates,
            'weight_decay': self.weight_decay,
            'max_epochs': self.max_epochs,
            'max_batches_per_epoch_training': self.max_batches_per_epochs_train,
            'max_batches_per_epoch_validation': self.max_batches_per_epochs_valid,
            'nb_cpu_processes': self.nb_cpu_processes,
            'use_gpu': self.use_gpu,
            'clip_grad': self.clip_grad,
            'comet_workspace': self.comet_workspace,
            'comet_project': self.comet_project,
            'optimizer': self.optimizer_key,
        }
        return params

    def save_params_to_json(self):
        """
        Save the trainer's parameters to a json file in the same folder as the
        experiment.
        """
        now = datetime.now()
        json_filename = os.path.join(self.saving_path, "parameters_{}.json"
                                     .format(now.strftime("%d-%m-%Y_%Hh%M")))
        with open(json_filename, 'w') as json_file:
            json_file.write(json.dumps(
                {'Date': str(datetime.now()),
                 'hdf5 file': self.batch_loader.dataset.hdf5_file,
                 'Trainer params': self.params_for_checkpoint,
                 'Sampler params': self.batch_sampler.params_for_checkpoint,
                 'Loader params': self.batch_loader.params_for_checkpoint,
                 'Model params': self.model.params_for_checkpoint,
                 'Patience': self.best_epoch_monitor.patience,
                 'Patience_delta': self.best_epoch_monitor.min_eps
                 },
                indent=4, separators=(',', ': ')))

        json_filename2 = os.path.join(self.saving_path,
                                      "parameters_latest.json")
        shutil.copyfile(json_filename, json_filename2)

    def save_checkpoint(self):
        """
        Saves an experiment checkpoint, with parameters and states.
        """
        logger.debug("Saving checkpoint...")
        checkpoint_dir = os.path.join(self.saving_path, "checkpoint")

        # Backup old checkpoint before saving, and erase it afterward.
        to_remove = None
        if os.path.exists(checkpoint_dir):
            to_remove = os.path.join(self.saving_path, "checkpoint_old")
            shutil.move(checkpoint_dir, to_remove)

        os.mkdir(checkpoint_dir)

        # Save experiment
        # Separated function to be re-implemented by child classes to fit your
        # needs. Below is one working example.
        checkpoint_state = self._prepare_checkpoint_info()
        torch.save(checkpoint_state,
                   os.path.join(checkpoint_dir, "checkpoint_state.pkl"))

        # Save model inside the checkpoint dir
        self.model.save_params_and_state(os.path.join(checkpoint_dir, 'model'))

        if to_remove:
            shutil.rmtree(to_remove)

    def _prepare_checkpoint_info(self) -> dict:
        """
        To instantiate a Trainer, we need the initialization parameters
        (self.params_for_checkpoint), and the states. This method returns
        the dictionary of required states.
        """
        # Note. batch sampler's rng state and batch loader's are the same.
        current_states = {
            # Rng value.
            'torch_rng_state': torch.random.get_rng_state(),
            'torch_cuda_state':
                torch.cuda.get_rng_state() if self.use_gpu else None,
            'sampler_np_rng_state': self.batch_sampler.np_rng.get_state(),
            'loader_np_rng_state': self.batch_loader.np_rng.get_state(),
            # Current epoch.
            'current_epoch': self.current_epoch,
            # Nb of batches per epoch.
            'nb_batches_train': self.nb_batches_train,
            'nb_batches_valid': self.nb_batches_valid,
            # Comet Experiment
            'comet_key': self.comet_key,
            # Optimizer
            'optimizer_state': self.optimizer.state_dict(),
        }

        # Monitors
        for monitor in self.monitors:
            current_states[monitor.name + '_state'] = monitor.get_state()

        # Additional params are the parameters necessary to load data, batch
        # samplers/loaders (see the example script dwiml_train_model.py).
        checkpoint_info = {
            'dataset_params': self.batch_sampler.dataset.params_for_checkpoint,
            'batch_sampler_params': self.batch_sampler.params_for_checkpoint,
            'batch_loader_params': self.batch_loader.params_for_checkpoint,
            'params_for_init': self.params_for_checkpoint,
            'current_states': current_states
        }

        return checkpoint_info

    @classmethod
    def init_from_checkpoint(
            cls, model: MainModelAbstract, experiments_path, experiment_name,
            batch_sampler: DWIMLBatchIDSampler,
            batch_loader: DWIMLStreamlinesBatchLoader,
            checkpoint_state: dict, new_patience, new_max_epochs, log_level):
        """
        Loads checkpoint information (parameters and states) to instantiate
        a Trainer. Current_epoch is updated +1.
        """
        trainer_params = checkpoint_state['params_for_init']

        trainer = cls(model=model, experiments_path=experiments_path,
                      experiment_name=experiment_name,
                      batch_sampler=batch_sampler,
                      batch_loader=batch_loader, from_checkpoint=True,
                      log_level=log_level, **trainer_params)

        if new_max_epochs:
            trainer.max_epochs = new_max_epochs

        # Save params to json to help user remember.
        current_states = checkpoint_state['current_states']
        trainer._update_states_from_checkpoint(current_states)

        if new_patience:
            trainer.best_epoch_monitor.patience = new_patience
        logger.info("Starting from checkpoint! Starting from epoch #{} "
                    "(i.e. #{}).\n"
                    "Previous best model was discovered at epoch {} (#{}).\n"
                    "When finishing previously, we were counting {} epochs "
                    "with no improvement."
                    .format(trainer.current_epoch,
                            trainer.current_epoch + 1,
                            trainer.best_epoch_monitor.best_epoch,
                            trainer.best_epoch_monitor.best_epoch + 1,
                            trainer.best_epoch_monitor.n_bad_epochs))

        return trainer

    @staticmethod
    def load_params_from_checkpoint(experiments_path: str,
                                    experiment_name: str):
        total_path = os.path.join(
            experiments_path, experiment_name, "checkpoint",
            "checkpoint_state.pkl")
        if not os.path.isfile(total_path):
            raise FileNotFoundError('Checkpoint was not found! ({})'
                                    .format(total_path))
        checkpoint_state = torch.load(total_path)

        return checkpoint_state

    def _update_states_from_checkpoint(self, current_states):
        """
        Updates all states from the checkpoint dictionary of states.
        """
        # A. Rng value.
        # RNG:
        #  - numpy
        self.batch_sampler.np_rng.set_state(current_states['sampler_np_rng_state'])
        self.batch_loader.np_rng.set_state(current_states['loader_np_rng_state'])
        #  - torch
        torch.set_rng_state(current_states['torch_rng_state'])
        if self.use_gpu:
            torch.cuda.set_rng_state(current_states['torch_cuda_state'])

        # B. Current epoch
        # Starting one epoch further than last time!
        self.current_epoch = current_states['current_epoch'] + 1

        # C. Nb of batches per epoch.
        self.nb_batches_train = current_states['nb_batches_train']
        self.nb_batches_valid = current_states['nb_batches_valid']

        # D. Comet Experiment
        # Experiment will be instantiated in train().
        self.comet_key = current_states['comet_key']

        # E. Optimizer
        self.optimizer.load_state_dict(current_states['optimizer_state'])

        # F. Monitors
        for monitor in self.monitors:
            monitor.set_state(current_states[monitor.name + '_state'])

    def _init_comet(self):
        """
        Initialize comet's experiment. User's account and workspace must be
        already set.
        """
        try:
            if self.comet_key:
                self.comet_exp = ExistingExperiment(
                    previous_experiment=self.comet_key)
            elif self.comet_workspace:
                if not self.comet_project:
                    raise ValueError("You have provided a comet workspace, "
                                     "but no comet project!")
                # New experiment
                self.comet_exp = CometExperiment(
                    project_name=self.comet_project,
                    workspace=self.comet_workspace,
                    log_code=False, log_graph=True, auto_param_logging=True,
                    auto_metric_logging=False, parse_args=False,
                    auto_output_logging='native', log_env_details=True,
                    log_env_gpu=True, log_env_cpu=True, log_env_host=False,
                    log_git_metadata=True, log_git_patch=True,
                    display_summary_level=False)
                self.comet_exp.set_name(self.experiment_name)
                self.comet_exp.log_parameters(self.params_for_checkpoint)
                self.comet_exp.log_parameters(self.batch_sampler.params_for_checkpoint)
                self.comet_exp.log_parameters(self.batch_loader.params_for_checkpoint)
                self.comet_exp.log_parameters(self.model.params_for_checkpoint)
                self.comet_exp.log_parameters(self.model.computed_params_for_display)
                self.comet_key = self.comet_exp.get_key()
                # Couldn't find how to set log level. Getting it directly.
                comet_log = logging.getLogger("comet_ml")
                comet_log.setLevel(logging.WARNING)
                comet_log = logging.getLogger("comet_ml.system.gpu.devices")
                comet_log.setLevel(logging.WARNING)

            elif self.comet_project:
                raise ValueError("You have provided a comet project, "
                                 "but no comet workspace!")
        except ConnectionError:
            logger.warning("Could not connect to Comet.ml, metrics will not "
                           "be logged online...")
            self.comet_exp = None
            self.comet_key = None

    def estimate_nb_batches_per_epoch(self):
        """
        Counts the number of training / validation batches required to see all
        the data (up to the maximum number of allowed batches).

        Data must be already loaded to access the information.
        """
        streamline_group = self.batch_sampler.streamline_group_idx
        train_set = self.batch_sampler.dataset.training_set
        valid_set = self.batch_sampler.dataset.validation_set

        nb_valid = 0
        if self.batch_sampler.batch_size_units == 'nb_streamlines':
            nb_train = train_set.total_nb_streamlines[streamline_group]
            if self.use_validation:
                nb_valid = valid_set.total_nb_streamlines[streamline_group]
        else:
            nb_train = train_set.total_nb_points[streamline_group]
            if self.use_validation:
                nb_valid = valid_set.total_nb_points[streamline_group]

        nb_train /= self.batch_sampler.batch_size_training
        final_nb_train = min(nb_train, self.max_batches_per_epochs_train)
        final_nb_train = max(final_nb_train, 1)  # Minimum 1 batch.

        if self.use_validation:  # Verifying or else, could divide by 0.
            nb_valid /= self.batch_sampler.batch_size_validation
            final_nb_valid = min(nb_valid, self.max_batches_per_epochs_valid)
            final_nb_valid = max(final_nb_valid, 1)
        else:
            final_nb_valid = 0

        return int(final_nb_train), int(final_nb_valid)

    def train_and_validate(self):
        """
        Trains + validates the model. Computes the training loss at each
        training loop, and many validation metrics at each validation loop.

        - Starts comet,
        - Creates DataLoaders from the BatchSamplers,
        - For each epoch
            - uses _train_one_epoch and _validate_one_epoch,
            - saves a checkpoint,
            - checks for earlyStopping if the loss is bad or patience is
              reached,
            - saves the model if the loss is good.
        """
        logger.debug("Trainer {}: \nRunning the model {}.\n\n"
                     .format(type(self), type(self.model)))
        self.save_params_to_json()

        # If data comes from checkpoint, this is already computed
        if self.nb_batches_train is None:
            logger.info("Estimating batch sizes.")
            (self.nb_batches_train, self.nb_batches_valid) = \
                self.estimate_nb_batches_per_epoch()

        # Instantiating comet
        self._init_comet()

        # Instantiating our IterTimer.
        # After each iteration, checks that the maximum allowed time has not
        # been reached.
        iter_timer = IterTimer(history_len=20)

        # Start from current_epoch in case the experiment is resuming
        # Train each epoch
        self.optimizer.zero_grad(set_to_none=True)
        for epoch in iter_timer(range(self.current_epoch, self.max_epochs)):
            # Updating current epoch. First epoch is 0!
            self.current_epoch = epoch
            if self.comet_exp:
                self.comet_exp.set_epoch(epoch)

            logger.info("\n\n******* STARTING : Epoch {} (i.e. #{}) *******"
                        .format(epoch, epoch + 1))

            # Computing learning rate
            current_lr = self.learning_rates[
                min(self.current_epoch, len(self.learning_rates) - 1)]
            logger.info("Learning rate = {}".format(current_lr))
            if self.comet_exp:
                self.comet_exp.log_metric("learning_rate", current_lr,
                                          step=epoch)

            for g in self.optimizer.param_groups:
                g['lr'] = current_lr

            # Training
            logger.info("*** TRAINING")
            self.train_one_epoch(epoch)

            # Validation
            if self.use_validation:
                logger.info("*** VALIDATION")
                self.validate_one_epoch(epoch)

            # Updating info
            mean_epoch_loss = self._get_latest_loss_to_supervise_best()
            is_bad = self.best_epoch_monitor.update(mean_epoch_loss, epoch)

            # Check if current best has been reached
            if self.comet_exp:
                # Saving this no matter what; we will see the stairs.
                self.comet_exp.log_metric(
                    "best_epoch", self.best_epoch_monitor.best_epoch)
            if is_bad:
                logger.info("** This is the {}th bad epoch (patience = {})"
                            .format(self.best_epoch_monitor.n_bad_epochs,
                                    self.best_epoch_monitor.patience))
            elif self.best_epoch_monitor.best_epoch == epoch:
                self._save_best_model()

            if self.comet_exp:
                self.comet_exp.log_metric(
                    "best_loss", self.best_epoch_monitor.best_value,
                    step=epoch)

            # End of epoch, save checkpoint for resuming later
            self.save_checkpoint()
            self.save_local_logs()

            # Check for early stopping
            if self.best_epoch_monitor.is_patience_reached:
                logger.warning(
                    "Early stopping! Loss has not improved after {} epochs!\n"
                    "Best result: {}; At epoch #{} (i.e. {})"
                    .format(self.best_epoch_monitor.patience,
                            self.best_epoch_monitor.best_value,
                            self.best_epoch_monitor.best_epoch,
                            self.best_epoch_monitor.best_epoch + 1))
                break

    def _get_latest_loss_to_supervise_best(self):
        """
        Defines the metric to be used to define the best model. Override if
        you have other validation metrics.
        """
        if self.use_validation:
            mean_epoch_loss = self.valid_local_loss_monitor.average_per_epoch[-1]
        else:
            mean_epoch_loss = self.train_loss_monitor.average_per_epoch[-1]

        return mean_epoch_loss

    def save_local_logs(self):
        """
        Save logs locally as numpy arrays.
        """
        for monitor in self.monitors:
            if isinstance(monitor, BatchHistoryMonitor):
                self._save_log_locally(monitor.average_per_epoch,
                                       monitor.name + '_per_epoch.npy')
            elif isinstance(monitor, TimeMonitor):
                self._save_log_locally(monitor.epoch_durations,
                                       monitor.name + '_duration.npy')

    def _save_log_locally(self, array: np.ndarray, fname: str):
        np.save(os.path.join(self.log_dir, fname), array)

    def _clear_handles(self):
        """
        Trying to improve the handles management.
        Todo. Improve again. CPU multiprocessing fails because of handles
         management.
        """
        # Make sure there are no existing HDF handles if using parallel workers
        if (self.nb_cpu_processes > 0 and
                self.batch_sampler.context_subset.is_lazy):
            self.batch_sampler.context_subset.close_all_handles()
            self.batch_sampler.context_subset.volume_cache_manager = None

    def back_propagation(self, loss):
        logger.debug('*** Computing back propagation')
        loss.backward()

        # Any other steps. Ex: clip gradients. Not implemented here.
        # See Learn2track's Trainer for an example.
        unclipped_grad_norm = self.fix_parameters()

        # Supervizing the gradient's norm.
        grad_norm = compute_gradient_norm(self.model.parameters())

        # Update parameters
        # Future work: We could update only every n steps.
        #  Effective batch size is n time bigger.
        #  See here https://towardsdatascience.com/optimize-pytorch-performance-for-speed-and-memory-efficiency-2022-84f453916ea6
        self.optimizer.step()

        # Reset parameter gradients to zero or to None before the next
        # forward pass
        self.optimizer.zero_grad(set_to_none=True)

        return unclipped_grad_norm, grad_norm

    def train_one_epoch(self, epoch):
        """
        Trains one epoch of the model: loops on all batches
        (forward + backpropagation).
        """
        for monitor in self.training_monitors:
            monitor.start_new_epoch()

        # Setting contexts
        self.batch_loader.set_context('training')
        self.batch_sampler.set_context('training')
        self.model.set_context('training')

        self._clear_handles()
        self.model.train()
        grad_context = torch.enable_grad

        # Training all batches
        # Note: loggers = [logging.root] only.
        # If we add our sub-loggers there, they duplicate.
        # A handler is added in the root logger, and sub-loggers propagate
        # their message.
        with tqdm_logging_redirect(self.train_dataloader, ncols=100,
                                   total=self.nb_batches_train,
                                   loggers=[logging.root],
                                   tqdm_class=tqdm) as pbar:

            train_iterator = enumerate(pbar)
            for batch_id, data in train_iterator:

                # Enable gradients for backpropagation. Uses torch's module
                # train(), which "turns on" the training mode.
                with grad_context():
                    mean_loss = self.train_one_batch(data)

                unclipped_grad_norm, grad_norm = self.back_propagation(
                    mean_loss)
                self.unclipped_grad_norm_monitor.update(unclipped_grad_norm)
                self.grad_norm_monitor.update(grad_norm)

                # Break if maximum number of batches has been reached
                if batch_id == self.nb_batches_train - 1:
                    # Explicitly breaking the loop here, else it calls the
                    # train_dataloader one more time, which samples a new
                    # batch that is not used (if we have not finished sampling
                    # all after nb_batches).
                    # Sending one more step to the tqdm bar, else it finishes
                    # at nb - 1.
                    pbar.update(1)

                    # Explicitly close tqdm's progress bar to fix possible bugs
                    # when breaking the loop
                    pbar.close()
                    break

            # Explicitly delete iterator to kill threads and free memory before
            # running validation
            del train_iterator

        # Saving epoch's information
        for monitor in self.training_monitors:
            monitor.end_epoch()
        self._update_comet_after_epoch('training', epoch)

        all_n = self.train_loss_monitor.current_epoch_batch_weights
        logger.info("Number of data points per batch: {}\u00B1{}"
                    .format(int(np.mean(all_n)), int(np.std(all_n))))

    def validate_one_epoch(self, epoch):
        """
        Validates one epoch of the model: loops on all batches.
        """
        for monitor in self.validation_monitors:
            monitor.start_new_epoch()

        # Setting contexts
        # Turn gradients off (no back-propagation)
        # Uses torch's module eval(), which "turns off" the training mode.
        self.batch_loader.set_context('validation')
        self.batch_sampler.set_context('validation')
        self.model.set_context('validation')
        self.model.eval()

        # Make sure there are no existing HDF handles if using parallel workers
        if (self.nb_cpu_processes > 0 and
                self.batch_sampler.context_subset.is_lazy):
            self.batch_sampler.context_subset.close_all_handles()

        # Validate all batches
        with tqdm_logging_redirect(self.valid_dataloader, ncols=100,
                                   total=self.nb_batches_valid,
                                   loggers=[logging.root],
                                   tqdm_class=tqdm) as pbar:
            valid_iterator = enumerate(pbar)
            for batch_id, data in valid_iterator:

                # Validate this batch: forward propagation + loss
                with torch.no_grad():
                    self.validate_one_batch(data, epoch)

                # Break if maximum number of epochs has been reached
                if batch_id == self.nb_batches_valid - 1:
                    # Explicitly breaking the loop here, else it calls the
                    # train_dataloader one more time, which samples a new
                    # batch that is not used (if we have not finished sampling
                    # all after nb_batches).
                    # Sending one more step to the tqdm bar, else it finishes
                    # at nb - 1.
                    pbar.update(1)

                    # Explicitly close tqdm's progress bar to fix possible bugs
                    # when breaking the loop
                    pbar.close()
                    break

            # Explicitly delete iterator to kill threads and free memory before
            # running training again
            del valid_iterator

        # Save info
        for monitor in self.validation_monitors:
            monitor.end_epoch()
        self._update_comet_after_epoch('validation', epoch)

    def train_one_batch(self, data):
        """
        Computes the loss for the current batch and updates monitors.
        Returns the loss to be used for backpropagation.
        """
        # Encapsulated for easier management of child classes.
        mean_local_loss, n = self.run_one_batch(data)

        # mean loss is a Tensor of a single value. item() converts to float
        self.train_loss_monitor.update(mean_local_loss.cpu().item(), weight=n)
        return mean_local_loss

    def validate_one_batch(self, data, epoch):
        """
        Computes the loss(es) for the current batch and updates monitors.
        """
        mean_local_loss, n = self.run_one_batch(data)
        self.valid_local_loss_monitor.update(mean_local_loss.cpu().item(),
                                             weight=n)

    def _update_comet_after_epoch(self, context: str, epoch: int):
        """
        Sends monitors information to comet.
        """
        if context == 'training':
            monitors = self.training_monitors
        elif context == 'validation':
            monitors = self.validation_monitors
        else:
            raise ValueError("Unexpected context ({}). Expecting "
                             "training or validation.")

        logs = []
        for monitor in monitors:
            if isinstance(monitor, BatchHistoryMonitor):
                value = monitor.average_per_epoch[-1]
            elif isinstance(monitor, TimeMonitor):
                value = monitor.epoch_durations[-1]
            else:
                continue
            logger.info("   Mean {} for this epoch: {}"
                        .format(monitor.name, value))
            logs.append((value, monitor.name))

        if self.comet_exp:
            # Comet context: will add train_(loss) or valid_(loss) to the
            # monitors name in comet.
            if context == 'training':
                comet_context = self.comet_exp.train
            else:  # context == 'validation':
                comet_context = self.comet_exp.validate

            with comet_context():
                # Not really implemented yet.
                # See https://github.com/comet-ml/issue-tracking/issues/247
                # Cheating. To have a correct plotting per epoch (no step)
                # using step = epoch. In comet_ml, it is intended to be
                # step = batch.
                for log in logs:
                    self.comet_exp.log_metric(
                        log[1], log[0], epoch=0, step=epoch)

    def _save_best_model(self):
        """
        Saves the current state of the model in the best_model folder.
        Saves the loss to a json folder.
        """
        logger.info("   Best epoch yet! Saving model and loss history.")

        # Save model
        self.model.save_params_and_state(
            os.path.join(self.saving_path, 'best_model'))

        best_losses = {
            'train_loss':
                self.train_loss_monitor.average_per_epoch[
                    self.best_epoch_monitor.best_epoch],
            'valid_loss':
                self.best_epoch_monitor.best_value if
                self.use_validation else None}
        with open(os.path.join(self.saving_path, "best_epoch_losses.json"),
                  'w') as json_file:
            json_file.write(json.dumps(best_losses, indent=4,
                                       separators=(',', ': ')))

    def run_one_batch(self, data):
        """
        Runs a batch of data through the model (calling its forward method)
        and returns the mean loss.

        Parameters
        ----------
        data : tuple of (List[StatefulTractogram], dict)
            Output of the batch loader's collate_fn.
            With our basic BatchLoader class, data is a tuple
            - batch_sfts: one sft per subject
            - final_streamline_ids_per_subj: the dict of streamlines ids from
              the list of all streamlines (if we concatenate all sfts'
              streamlines)
        n: int
            The number of points in this batch
        X: Any
            Any other data returned when computing loss. Not used in the
            trainer, but could be useful anywhere else.
        """
        # Data interpolation has not been done yet. GPU computations are done
        # here in the main thread.
        targets, ids_per_subj, data_per_streamline = data

        # Dataloader always works on CPU. Sending to right device.
        # (model is already moved).
        targets = [s.to(self.device, non_blocking=True, dtype=torch.float)
                   for s in targets]

        # Uses the model's method, with the batch_loader's data.
        # Possibly skipping the last point if not useful.
        streamlines_f = targets

        # Possibly add noise to inputs here.
        logger.debug('*** Computing forward propagation')

        # Now possibly add noise to streamlines (training / valid)
        streamlines_f = self.batch_loader.add_noise_streamlines_forward(
            streamlines_f, self.device)

        # Possibly computing directions twice (during forward and loss)
        # but ok, shouldn't be too heavy. Easier to deal with multiple
        # projects' requirements by sending whole streamlines rather
        # than only directions.
        model_outputs = self.model(streamlines_f, data_per_streamline)
        del streamlines_f

        logger.debug('*** Computing loss')
        targets = self.batch_loader.add_noise_streamlines_loss(
            targets, self.device)

        results = self.model.compute_loss(model_outputs, targets,
                                          average_results=True)

        if self.use_gpu:
            log_gpu_memory_usage(logger)

        # The mean tensor is a single value. Converting to float using item().
        return results

    def fix_parameters(self):
        """
        This function is called during training, after the forward and
        backward propagation, but before updating the parameters through the
        optimizer. User may define their own functions here if some
        modification on the parameters is necessary.

        Here: clipping gradient, to avoid exploding gradients problem.
        """
        if self.clip_grad is not None:
            unclipped_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_grad)
        else:
            unclipped_grad_norm = compute_gradient_norm(
                self.model.parameters())
        if torch.isnan(unclipped_grad_norm):
            raise ValueError("Exploding gradients. Experiment failed.")

        return unclipped_grad_norm.cpu().numpy()

    @staticmethod
    def check_stopping_cause(checkpoint_state, new_patience=None,
                             new_max_epochs=None):
        """
        This method should be used before starting the training. Verifies that
        it makes sense to continue training based on number of epochs and
        patience.
        """
        current_epoch = checkpoint_state['current_states']['current_epoch']

        # 1. Check if early stopping had been triggered.
        best_monitor_state = \
            checkpoint_state['current_states']['best_epoch_monitor_state']
        bad_epochs = best_monitor_state['n_bad_epochs']
        if new_patience is None:
            # No new patience: checking if early stopping had been triggered.
            if bad_epochs >= best_monitor_state['patience']:
                raise EarlyStoppingError(
                    "Resumed experiment was stopped because of early "
                    "stopping (patience {} reached at epcoh {}).\n"
                    "Increase patience in order to resume training!"
                    .format(best_monitor_state['patience'], current_epoch))
        elif bad_epochs >= new_patience:
            # New patience: checking if we will be able to continue
            raise EarlyStoppingError(
                "In resumed experiment, we had reach {} bad epochs (i.e. with "
                "no improvement). You have now overriden patience to {} but "
                "that won't be enough. Please increase that value in "
                "order to resume training."
                .format(best_monitor_state['n_bad_epochs'], new_patience))

        # 2. Checking that max_epochs had not been reached.
        if new_max_epochs is None:
            if current_epoch == \
                    checkpoint_state['params_for_init']['max_epochs'] - 1:
                raise ValueError(
                    "Resumed experiment had stopped after reaching the "
                    "maximum number of epochs allowed (max_epochs = {}).\n"
                    "Please increase that value in order to resume training."
                    .format(checkpoint_state['params_for_init']['max_epochs']))
        else:
            if current_epoch + 1 >= new_max_epochs:
                raise ValueError(
                    "In resumed experiment, we had performed {} epochs. \nYou "
                    "have now overriden max_epoch to {} but that won't be "
                    "enough. \nPlease increase that value in order to resume "
                    "training.".format(current_epoch + 1, new_max_epochs))


class DWIMLTrainerOneInput(DWIMLAbstractTrainer):
    batch_loader: DWIMLBatchLoaderOneInput

    def run_one_batch(self, data):
        """
        Run a batch of data through the model (calling its forward method)
        and return the mean loss. If training, run the backward method too.

        Parameters
        ----------
        data : tuple of (List[StatefulTractogram], dict)
            This is the output of the AbstractBatchLoader's
            load_batch_streamlines()
            method. data is a tuple
            - batch_sfts: one sft per subject
            - final_streamline_ids_per_subj: the dict of streamlines ids from
              the list of all streamlines (if we concatenate all sfts'
              streamlines).

        Returns
        -------
        mean_loss : Tensor of shape (1,) ; float.
            The mean loss of the provided batch.
        n: int
            Total number of points for this batch.
        """
        # Data interpolation has not been done yet. GPU computations are done
        # here in the main thread.
        targets, ids_per_subj, data_per_streamline = data

        # Dataloader always works on CPU. Sending to right device.
        # (model is already moved).
        targets = [s.to(self.device, non_blocking=True, dtype=torch.float)
                   for s in targets]

        # Getting the inputs points from the volumes.
        # Uses the model's method, with the batch_loader's data.
        # Possibly skipping the last point if not useful.
        streamlines_f = targets
        if isinstance(self.model, ModelWithDirectionGetter) and \
                not self.model.direction_getter.add_eos:
            # No EOS = We don't use the last coord because it does not have an
            # associated target direction.
            streamlines_f = [s[:-1, :] for s in streamlines_f]

        # Batch inputs is already the right length. Models don't need to
        # discard the last point if no EOS. Avoid interpolation for no reason.
        batch_inputs = self.batch_loader.load_batch_inputs(
            streamlines_f, ids_per_subj)

        logger.debug('*** Computing forward propagation')
        # todo Possibly add noise to inputs here. Not ready
        # Now add noise to streamlines for the forward pass
        # (batch loader will do it depending on training / valid)
        streamlines_f = self.batch_loader.add_noise_streamlines_forward(
            streamlines_f, self.device)
        model_outputs = self.model(
            batch_inputs, streamlines_f, data_per_streamline)
        del streamlines_f

        logger.debug('*** Computing loss')
        # Add noise to targets.
        # (batch loader will do it depending on training / valid)
        targets = self.batch_loader.add_noise_streamlines_loss(targets,
                                                               self.device)
        mean_loss, n = self.model.compute_loss(model_outputs, targets,
                                               average_results=True)

        if self.use_gpu:
            log_gpu_memory_usage(logger)

        return mean_loss, n
