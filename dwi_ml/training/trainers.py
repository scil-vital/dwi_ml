# -*- coding: utf-8 -*-
import json
import logging
import os
import shutil
import time
from typing import Union

from comet_ml import (Experiment as CometExperiment, ExistingExperiment)
import contextlib2
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from dwi_ml.experiment_utils.memory import log_gpu_memory_usage
from dwi_ml.experiment_utils.prints import (
    make_logger_tqdm_fitted, make_logger_normal)
from dwi_ml.models.main_models import MainModelAbstract
from dwi_ml.training.batch_loaders import (
    AbstractBatchLoader, BatchLoaderOneInput)
from dwi_ml.training.batch_samplers import DWIMLBatchSampler
from dwi_ml.training.gradient_norm import compute_gradient_norm
from dwi_ml.training.monitoring import (
    BestEpochMonitoring, EarlyStoppingError, IterTimer, ValueHistoryMonitor)

# If the remaining time is less than one epoch + X seconds, we will quit
# training now, to allow updating taskman_report.
QUIT_TIME_DELAY_SECONDS = 30
logger = logging.getLogger('train_logger')


class DWIMLAbstractTrainer:
    """
    This Trainer class's train_and_validate() method:
        - Creates DataLoaders from the data_loaders. Collate_fn will be the
        loader.load_batch() method, and the dataset will be
        sampler.source_data.
        - Trains each epoch by using compute_batch_loss, which should be
        implemented in each project's child class.

    Comet is used to save training information, but some logs will also be
    saved locally in the saving_path.
    """
    def __init__(self,
                 model: MainModelAbstract, experiments_path: str,
                 experiment_name: str,
                 batch_sampler_training: DWIMLBatchSampler,
                 batch_loader_training: AbstractBatchLoader,
                 batch_sampler_validation: DWIMLBatchSampler = None,
                 batch_loader_validation: AbstractBatchLoader = None,
                 model_uses_streamlines: bool = False,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.01, max_epochs: int = 10,
                 max_batches_per_epoch: int = 1000, patience: int = None,
                 nb_cpu_processes: int = 0, taskman_managed: bool = False,
                 use_gpu: bool = False, comet_workspace: str = None,
                 comet_project: str = None, from_checkpoint: bool = False,
                 log_level=logging.root.level):
        """
        Parameters
        ----------
        model: MainModelAbstract
            Instatiated class containing your model.
        experiments_path: str
            Path where to save this experiment's results and checkpoints.
            Will be saved in experiments_path/experiment_name.
        experiment_name: str
            Name of this experiment. This will also be the name that will
            appear online for comet.ml experiment.
        batch_sampler_training: DWIMLBatchSampler
            Instantiated class used for sampling batches of training data.
            Data in batch_sampler_training.source_data must be already loaded.
        batch_loader_training: AbstractBatchLoader
            Instantiated class with a load_batch method able to load data
            associated to sampled batch ids.
        batch_sampler_validation: DWIMLBatchSampler
            Similar as before, for the validation set. Can be set to None if no
            validation is used. Then, best model is based on training loss.
        batch_loader_validation: AbstractBatchLoader
            Again, similar as before but can be set to None.
        model_uses_streamlines: bool
            If true, the batch streamlines will be sent to the model when
            calling the forward method. Else, only the inputs. Default: False.
        learning_rate: float
            Learning rate. Default: 0.001 (torch's default)
        weight_decay: float
            Add a weight decay penalty on the parameters. Default: 0.01.
            (torch's default).
        max_epochs: int
            Maximum number of epochs. Default = 10, for no good reason.
        max_batches_per_epoch: int
            Maximum number of batches per epoch. Default = 10000, for no good
            reason.
        patience: int
            Use early stopping. Defines the number of epochs after which the
            model should stop if the loss hasn't improved. Default: None (i.e.
            no early stopping).
        nb_cpu_processes: int
            Number of parallel CPU workers. Use 0 to avoid parallel threads.
            Default : 0.
        taskman_managed: bool
            If True, taskman manages the experiment. Do not output progress
            bars and instead output special messages for taskman.
            Default: False (i.e. use tqdm).
        use_gpu: bool
            If true, use GPU device when possible instead of CPU.
            Default = False
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

        # Trainer's logging level can be changed separately from main
        # scripts.
        self.logger = logger
        self.logger.setLevel(log_level)

        # Experiment
        if not os.path.isdir(experiments_path):
            raise NotADirectoryError("The experiments path does not exist! "
                                     "({})".format(experiments_path))

        self.experiments_path = experiments_path
        self.saving_path = os.path.join(experiments_path,
                                        experiment_name)
        if not from_checkpoint and not os.path.isdir(self.saving_path):
            logging.info('Creating directory {}'.format(self.saving_path))
            os.mkdir(self.saving_path)

        self.experiment_name = experiment_name
        self.saving_path = os.path.join(self.experiments_path,
                                        self.experiment_name)
        if not from_checkpoint and not os.path.isdir(self.saving_path):
            logger.info('Creating directory {}'.format(self.saving_path))
            os.mkdir(self.saving_path)

        # Note that the training/validation sets are contained in the
        # data_loaders.data_source
        self.train_batch_sampler = batch_sampler_training
        self.valid_batch_sampler = batch_sampler_validation
        if self.valid_batch_sampler is None:
            self.use_validation = False
            self.logger.warning(
                "WARNING! There is not validation set. Loss for best epoch "
                "monitoring will be the training loss. \n"
                "Best practice is to have a validation set.")
        else:
            self.use_validation = True
        self.train_batch_loader = batch_loader_training
        self.valid_batch_loader = batch_loader_validation
        self.model = model
        self.model_uses_streamlines = model_uses_streamlines
        self.max_epochs = max_epochs
        self.max_batches_per_epochs = max_batches_per_epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.nb_cpu_processes = nb_cpu_processes
        self.taskman_managed = taskman_managed
        self.use_gpu = use_gpu

        self.comet_workspace = comet_workspace
        self.comet_project = comet_project

        # ----------------------
        # Values fixed by us
        # ----------------------

        # Device and rng value. Note that if loading from a checkpoint, the
        # complete state should be updated.
        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')

                # Setting the rng seed
                if (self.use_validation and
                        self.train_batch_sampler.rng !=
                        self.valid_batch_sampler.rng):
                    raise ValueError("Training and validation batch samplers "
                                     "do not have the same rng. Please verify "
                                     "the code.")
                # If you see a hint error below, upgrade torch.
                torch.cuda.manual_seed(self.train_batch_sampler.rng)
            else:
                raise ValueError("You chose GPU (cuda) device but it is not "
                                 "available!")
        else:
            self.device = torch.device('cpu')

        # ----------------------
        # Values that will be modified later on. If initializing experiment
        # from a checkpoint, these values should be updated after
        # initialization.
        # ----------------------
        if patience:
            self.best_epoch_monitoring = BestEpochMonitoring(
                patience=self.patience)
        else:
            # We won't use early stopping to stop the epoch, but we will use
            # it as monitor of the best epochs.
            self.best_epoch_monitoring = BestEpochMonitoring(
                patience=self.max_batches_per_epochs + 1)

        self.current_epoch = 0

        # Nb of batches with be estimated later on
        self.nb_train_batches_per_epoch = None
        self.nb_valid_batches_per_epoch = None

        self.taskman_report = {
            'loss_train': None,
            'loss_valid': None,
            'epoch': None,
            'best_epoch': None,
            'best_score': None,
            'update': None,
            'update_loss': None,
            'time': 0.
        }

        # RNG state
        # Nothing to to here.

        # Setup monitors
        self.train_loss_monitor = ValueHistoryMonitor("Training loss")
        self.valid_loss_monitor = ValueHistoryMonitor("Validation loss")
        self.grad_norm_monitor = ValueHistoryMonitor("Grad Norm")

        # Comet values will be instantiated in train().
        self.comet_exp = None
        self.comet_key = None

        # ----------------------
        # Launching optimizer!
        # ----------------------

        # Prepare optimizer
        # Send model to device. Reminder, contrary to tensors, model.to
        # overwrites the model.
        # NOTE: This ordering is important! The optimizer needs to use the cuda
        # Tensors if using the GPU...
        self.model.to(device=self.device)

        # Build optimizer (Optimizer is built here since it needs the model
        # parameters)
        list_params = [n for n, _ in self.model.named_parameters()]
        self.logger.debug("Initiating trainer: {}".format(type(self)))
        self.logger.debug("This trainer will use Adam optimization on the "
                          "following model.parameters: \n\n"
                          .join(list_params) + "\n")
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate,
                                          weight_decay=weight_decay)

    @property
    def params_for_checkpoint(self):
        # These are the parameters necessary to use _init_
        params = {
            'model_uses_streamlines': self.model_uses_streamlines,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'max_epochs': self.max_epochs,
            'max_batches_per_epoch': self.max_batches_per_epochs,
            'patience': self.patience,
            'nb_cpu_processes': self.nb_cpu_processes,
            'taskman_managed': self.taskman_managed,
            'use_gpu': self.use_gpu,
            'comet_workspace': self.comet_workspace,
            'comet_project': self.comet_project
        }
        return params

    @property
    def params(self) -> dict:
        params = self.params_for_checkpoint
        params.update({
            'experiments_path': self.experiments_path,
            'experiment_name': self.experiment_name,
            'comet_key': self.comet_key,
            'computed_values': {
                'nb_training_batches_per_epoch':
                    self.nb_train_batches_per_epoch,
                'nb_validation_batches_per_epoch':
                    self.nb_valid_batches_per_epoch
            }
            })
        return params

    def _init_comet(self):
        """
        For more information on comet, see our doc/Getting Started
        """
        try:
            if self.comet_key:
                self.comet_exp = ExistingExperiment(
                    previous_experiment=self.comet_key)
            elif self.comet_workspace:
                # New experiment
                # Use trainset name as comet project name
                project_name = self.comet_project
                self.comet_exp = CometExperiment(
                    project_name=project_name, workspace=self.comet_workspace,
                    log_code=False, log_graph=True, auto_param_logging=True,
                    auto_metric_logging=False, parse_args=False,
                    auto_output_logging='native', log_env_details=True,
                    log_env_gpu=True, log_env_cpu=True, log_env_host=False,
                    log_git_metadata=True, log_git_patch=True,
                    display_summary=False)
                self.comet_exp.set_name(self.experiment_name)
                self.comet_exp.log_parameters(self.params)
                self.comet_key = self.comet_exp.get_key()
        except ConnectionError:
            self.logger.warning(
                "Could not connect to Comet.ml, metrics will not be logged "
                "online...")
            self.comet_exp = None
            self.comet_key = None

    def estimate_nb_batches_per_epoch(self):
        """
        Please override in your child class if you have a better way to
        define the epochs sizes.

        Returns:
             (nb_training_batches_per_epoch, nb_validation_batches_per_epoch)
        """
        return self.max_batches_per_epochs, self.max_batches_per_epochs

    def train_and_validate(self, *args):
        """
        Train + validates the model (+ computes loss)

        - Starts comet,
        - Creates DataLoaders from the BatchSamplers,
        - For each epoch
            - uses _train_one_epoch and _validate_one_epoch,
            - checks for earlyStopping if the loss is bad,
            - saves the model if the loss is good.
        - Checks if allowed training time is exceeded.

        Parameters
        ----------
        All *args will be passed all the way to _train_one_epoch and
        _train_one_batch, in case you want to override them.
        """
        self.logger.debug("Trainer {}: \n"
                          "Running the model {}.\n\n"
                          .format(type(self), type(self.model)))

        # If data comes from checkpoint, this is already computed
        if self.nb_train_batches_per_epoch is None:
            self.logger.info("Estimating batch sizes.")
            (self.nb_train_batches_per_epoch,
             self.nb_valid_batches_per_epoch) = \
                self.estimate_nb_batches_per_epoch()

        # Instantiate comet experiment
        # If self.comet_key is None: new experiment, will create a key
        # Else, resuming from checkpoint. Will continue with given key.
        self._init_comet()
        if self.comet_exp:
            train_context = self.comet_exp.train_and_validate
            valid_context = self.comet_exp.validate
        else:
            # Instantiating contexts doing nothing instead
            train_context = contextlib2.nullcontext
            valid_context = contextlib2.nullcontext

        # Create DataLoaders from the BatchSamplers
        #   * Pin memory if interpolation is done by workers; this means that
        #     dataloader output is on GPU, ready to be fed to the model.
        #     Otherwise, dataloader output is kept on CPU, and the main thread
        #     sends volumes and coords on GPU for interpolation.
        self.logger.debug("- Instantiating dataloaders...")

        # toDo We wouldn't need training / valid batch samplers and loaders if
        #  I knew how to add option 'training' and 'validation' to the
        #  __iter__ method or to the collate_fn (load_batch). But maybe the
        #  user wants separate options. During validation and training. Ex:
        #  less on-the-fly noise addition to the streamlines during validation?
        #  But I don't see why we wouldn't want the same batch sampler. We
        #  could have only one and use the same.copy() and change the value of
        #  the subset to training or validation.
        #  If we also don't think users want different load_batch, solution
        #  could be (for the dataloader) the collate_fn could be nothing, and
        #  we call load_data() ourselves after, with options.

        train_dataloader = DataLoader(
            self.train_batch_sampler.dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=self.nb_cpu_processes,
            collate_fn=self.train_batch_loader.load_batch,
            pin_memory=self.use_gpu)

        valid_dataloader = None
        if self.use_validation:
            valid_dataloader = DataLoader(
                self.valid_batch_sampler.dataset,
                batch_sampler=self.valid_batch_sampler,
                num_workers=self.nb_cpu_processes,
                collate_fn=self.valid_batch_loader.load_batch,
                pin_memory=self.use_gpu)

        # Instantiating our IterTimer.
        # After each iteration, checks that the maximum allowed time has not
        # been reached.
        iter_timer = IterTimer(history_len=20)

        # Start from current_epoch in case the experiment is resuming
        # Train each epoch
        for epoch in iter_timer(range(self.current_epoch, self.max_epochs)):
            # Updating current epoch. First epoch is 0!
            self.current_epoch = epoch

            # Training
            self.logger.info("**********TRAINING: Epoch #{}*************"
                             .format(epoch))
            self.train_one_epoch(train_dataloader, train_context, epoch)

            # Validation
            if self.use_validation:
                self.logger.info("**********VALIDATION: Epoch #{}*************"
                                 .format(epoch))
                self.validate_one_epoch(valid_dataloader, valid_context, epoch,
                                        *args)

                last_loss = self.valid_loss_monitor.epochs_means_history[-1]
            else:
                last_loss = self.train_loss_monitor.epochs_means_history[-1]

            # Updating info
            self.best_epoch_monitoring.update(last_loss, epoch)

            # Check for early stopping
            if self.best_epoch_monitoring.is_patience_reached:
                self.save_checkpoint()
                raise EarlyStoppingError(
                    "Early stopping! Loss has not improved after {} epochs!\n"
                    "Best result: {}; At epoch #{}"
                    .format(self.patience,
                            self.best_epoch_monitoring.best_value,
                            self.best_epoch_monitoring.best_epoch))

            # Else, check if current best has been reached
            # If that is the case, the monitor has just reset its n_bad_epochs
            # to 0
            if self.best_epoch_monitoring.n_bad_epochs == 0:
                self.logger.info("Best epoch yet! Saving model and loss "
                                 "history.")

                # Save model
                self.model.update_best_model()
                self.model.save(self.saving_path)

                # Save losses (i.e. mean over all batches)
                losses = {
                    'train_loss':
                        self.train_loss_monitor.epochs_means_history[
                            self.best_epoch_monitoring.best_epoch],
                    'valid_loss':
                        self.best_epoch_monitoring.best_value if
                        self.use_validation else None}
                with open(os.path.join(self.saving_path, "losses.json"),
                          'w') as json_file:
                    json_file.write(json.dumps(losses, indent=4,
                                               separators=(',', ': ')))

                # Save information online
                if self.comet_exp:
                    self.comet_exp.log_metric(
                        "best_loss",
                        self.best_epoch_monitoring.best_value)
                    self.comet_exp.log_metric(
                        "best_epoch",
                        self.best_epoch_monitoring.best_epoch)

            # End of epoch, save checkpoint for resuming later
            self.save_checkpoint()
            if self.taskman_managed:
                updates = {
                    'loss_train':
                        self.train_loss_monitor.epochs_means_history[-1],
                    'loss_valid':
                        self.valid_loss_monitor.epochs_means_history[-1] if
                        self.use_validation else None,
                    'epoch': self.current_epoch,
                    'best_epoch': self.best_epoch_monitoring.best_epoch,
                    'best_loss': self.best_epoch_monitoring.best_value
                }
                self._update_taskman_report(updates)

    def save_model(self):
        self.model.save(self.saving_path)

    def train_one_epoch(self, train_dataloader, train_context, epoch):
        """
        Train one epoch of the model: loop on all batches.

        All *args will be passed all to run_one_batch, which you should
        implement, in case you need some variables.
        """
        # Make sure there are no existing HDF handles if using parallel workers
        if (self.nb_cpu_processes > 0 and
                self.train_batch_sampler.dataset.is_lazy):
            self.train_batch_sampler.dataset.hdf_handle = None
            self.train_batch_sampler.dataset.volume_cache_manager = None

        if self.comet_exp:
            self.comet_exp.log_metric("current_epoch", self.current_epoch)

        # Improving loggers for tqdm
        make_logger_tqdm_fitted(self.logger)
        make_logger_tqdm_fitted(self.model.logger)
        make_logger_tqdm_fitted(self.train_batch_sampler.logger)
        make_logger_tqdm_fitted(self.train_batch_loader.logger)
        if self.valid_batch_sampler:
            make_logger_tqdm_fitted(self.valid_batch_sampler.logger)
            make_logger_tqdm_fitted(self.valid_batch_loader.logger)

        # Training all batches
        self.logger.debug("Training one epoch: iterating on batches using "
                          "tqdm on the dataloader...")
        with tqdm(train_dataloader, ncols=100, disable=self.taskman_managed,
                  total=self.nb_train_batches_per_epoch) as pbar:
            train_iterator = enumerate(pbar)
            with train_context():
                for batch_id, data in train_iterator:
                    # Break if maximum number of epochs has been reached
                    if batch_id == self.nb_train_batches_per_epoch:
                        # Explicitly close tqdm's progress bar to fix possible
                        # bugs when breaking the loop
                        pbar.close()
                        break

                    mean_loss, grad_norm = self.run_one_batch(
                        data, is_training=True,
                        batch_loader=self.train_batch_loader)

                    self.logger.debug("Updated loss: {}".format(mean_loss))
                    self.train_loss_monitor.update(mean_loss)
                    self.grad_norm_monitor.update(grad_norm)

                    # Update information every 10 updates
                    if not self.use_validation and batch_id % 10 == 0:
                        self._update_logs(batch_id, mean_loss)

            # Explicitly delete iterator to kill threads and free memory before
            # running validation
            del train_iterator

        # Making loggers normal
        make_logger_normal(self.logger)
        make_logger_normal(self.model.logger)
        make_logger_normal(self.train_batch_sampler.logger)
        make_logger_normal(self.train_batch_loader.logger)
        if self.valid_batch_sampler:
            make_logger_normal(self.valid_batch_sampler.logger)
            make_logger_normal(self.valid_batch_loader.logger)

        # Saving epoch's information
        self.logger.info("Finishing epoch...")
        self.train_loss_monitor.end_epoch()
        self.grad_norm_monitor.end_epoch()
        self._save_log_from_array(self.train_loss_monitor.epochs_means_history,
                                  "train_loss.npy")
        self._save_log_from_array(self.grad_norm_monitor.epochs_means_history,
                                  "gradient_norm.npy")
        with train_context():
            if self.comet_exp:
                self.comet_exp.log_metric(
                    "gradient_norm_epoch",
                    self.grad_norm_monitor.epochs_means_history[-1],
                    step=epoch)
                self.comet_exp.log_metric(
                    "loss_epoch",
                    self.train_loss_monitor.epochs_means_history[-1],
                    step=epoch)

        self.logger.info(
            "Mean gradient norm : {}"
            .format(self.grad_norm_monitor.epochs_means_history[-1]))
        self.logger.info(
            "Mean training loss : {}"
            .format(self.train_loss_monitor.epochs_means_history[-1]))

    def validate_one_epoch(self, valid_dataloader, valid_context, epoch,
                           *args):
        """
        Validate one epoch of the model: loop on all batches.

        All *args will be passed all to run_one_batch, which you should
        implement, in case you need some variables.
        """
        self.logger.debug('Unused args in validate: {}'.format(args))

        # Make sure there are no existing HDF handles if using parallel workers
        if (self.nb_cpu_processes > 0 and
                self.valid_batch_sampler.dataset.is_lazy):
            self.valid_batch_sampler.dataset.hdf_handle = None
            self.valid_batch_sampler.dataset.volume_cache_manager = None

        # Validate all batches
        with tqdm(valid_dataloader, ncols=100, disable=self.taskman_managed,
                  total=self.nb_valid_batches_per_epoch) as pbar:
            valid_iterator = enumerate(pbar)
            for batch_id, data in valid_iterator:
                # Break if maximum number of epochs has been reached
                if batch_id == self.nb_valid_batches_per_epoch:
                    # Explicitly close tqdm's progress bar to fix possible bugs
                    # when breaking the loop
                    pbar.close()
                    break

                # Validate this batch: forward propagation + loss
                mean_loss, _ = self.run_one_batch(
                    data, is_training=False,
                    batch_loader=self.valid_batch_loader)
                self.valid_loss_monitor.update(mean_loss)

                # Update information every 10 updates
                if batch_id % 10 == 0:
                    self._update_logs(batch_id, mean_loss)

            # Explicitly delete iterator to kill threads and free memory before
            # running training again
            del valid_iterator

        # Save this epoch's information
        self.valid_loss_monitor.end_epoch()
        self._save_log_from_array(self.valid_loss_monitor.epochs_means_history,
                                  "valid_loss.npy")
        with valid_context():
            if self.comet_exp:
                self.comet_exp.log_metric(
                    "loss_epoch",
                    self.valid_loss_monitor.epochs_means_history[-1],
                    step=epoch)
        self.logger.info(
            "Validation loss : {}"
            .format(self.valid_loss_monitor.epochs_means_history[-1]))

    def _update_logs(self, batch_id, mean_loss):
        if self.taskman_managed:
            th = self.train_loss_monitor.epochs_means_history
            vh = self.valid_loss_monitor.epochs_means_history if \
                self.use_validation else []
            updates = {
                'loss_train': th[-1] if len(th) > 0 else None,
                'loss_valid': vh[-1] if len(vh) > 0 else None,
                'epoch': self.current_epoch,
                'best_epoch': self.best_epoch_monitoring.best_epoch,
                'best_loss': self.best_epoch_monitoring.best_value,
                'update': batch_id,
                'update_loss': mean_loss
            }
            self._update_taskman_report(updates)

        if self.comet_exp:
            self.comet_exp.log_metric("loss_step", mean_loss, step=batch_id)
            self.comet_exp.log_metric(
                "gradient_norm_step",
                self.grad_norm_monitor.current_epoch_history[-1],
                step=batch_id)

    def run_one_batch(self, data, is_training: bool, batch_loader):
        """
        Run a batch of data through the model (calling its forward method)
        and return the mean loss. If training, run the backward method too.

        If the sampler was instantiated with wait_for_gpu, then we need to
        compute the inputs here; not done yet.

        Parameters
        ----------
        data : tuple of (List, dict)
            This is the output of the AbstractBatchLoader's load_batch()
            method. If wait_for_gpu, data is
            (batch_streamlines, final_streamline_ids_per_subj). Else, data is
            (batch_streamlines, final_streamline_ids_per_subj, inputs)
        batch_loader: AbstractBatchLoader
            Either self.train_batch_loader or valid_batch_loader, depending
            on the case.
        is_training : bool
            If True, record the computation graph and backprop through the
            model parameters.
        Returns
        -------
        mean_loss : float
            The mean loss of the provided batch
        grad_norm: float
            The total norm (sqrt(sum(params**2))) of parameters before gradient
            clipping, if any.
        """
        raise NotImplementedError

    def compute_loss(self, model_outputs, targets):
        """
        Calls the compute_loss method of the model. Reimplement in a child
        class if targets needs to be formatted in any way before the call.
        """
        mean_loss = self.model.compute_loss(model_outputs, targets)
        return mean_loss

    def fix_parameters(self):
        """
        This function is called during training, after the forward and
        backward propagation, but before updating the parameters through the
        optimizer. User may define their own functions here if some
        modification on the parameters is necessary.
        Ex: in the case of vanishing or exploding gradients problem, this would
        be the place to fix the parameters based on the gradient.
        """
        pass

    @classmethod
    def init_from_checkpoint(
            cls, model: MainModelAbstract, experiments_path, experiment_name,
            train_batch_sampler: DWIMLBatchSampler,
            train_batch_loader: AbstractBatchLoader,
            valid_batch_sampler: Union[DWIMLBatchSampler, None],
            valid_batch_loader: Union[AbstractBatchLoader, None],
            checkpoint_state: dict, new_patience,
            new_max_epochs):
        """
        During save_checkpoint(), checkpoint_state.pkl is saved. Loading it
        back offers a dict that can be used to instantiate an experiment and
        set it at the same state as previously. (Current_epoch is updated +1).

        Hint: If you want to use this in your child class, use:
        experiment, checkpoint_state = super(cls, cls).init_from_checkpoint(...
        """
        trainer = cls(model, experiments_path, experiment_name,
                      batch_sampler_training=train_batch_sampler,
                      batch_loader_training=train_batch_loader,
                      batch_sampler_validation=valid_batch_sampler,
                      batch_loader_validation=valid_batch_loader,
                      from_checkpoint=True,
                      **checkpoint_state['params_for_init'])

        current_states = checkpoint_state['current_states']

        # Overriding values
        if new_patience:
            trainer.patience = new_patience
        if new_max_epochs:
            trainer.max_epochs = new_max_epochs

        # Set RNG states
        torch.set_rng_state(current_states['torch_rng_state'])
        trainer.train_batch_sampler.np_rng.set_state(
            current_states['numpy_rng_state'])
        if trainer.use_validation:
            trainer.valid_batch_sampler.np_rng.set_state(
                current_states['numpy_rng_state'])
        if trainer.use_gpu:
            torch.cuda.set_rng_state(current_states['torch_cuda_state'])

        # Set other objects
        trainer.comet_key = current_states['comet_key']
        trainer.current_epoch = current_states['current_epoch'] + 1
        trainer.nb_train_batches_per_epoch = \
            current_states['nb_train_batches_per_epoch']
        trainer.nb_valid_batches_per_epoch = \
            current_states['nb_valid_batches_per_epoch']
        trainer.best_epoch_monitoring.set_state(
            current_states['best_epoch_monitoring_state'])
        trainer.train_loss_monitor.set_state(
            current_states['train_loss_monitor_state'])
        trainer.valid_loss_monitor.set_state(
            current_states['valid_loss_monitor_state'])
        trainer.grad_norm_monitor.set_state(
            current_states['grad_norm_monitor_state'])
        trainer.optimizer.load_state_dict(current_states['optimizer_state'])

        logger.info("Resuming from checkpoint! Next epoch will be epoch #{}"
                    .format(trainer.current_epoch))

        return trainer

    def save_checkpoint(self):
        """
        Save an experiment checkpoint that can be resumed from.
        """
        self.logger.info("Saving checkpoint...")

        # Make checkpoint directory
        checkpoint_dir = os.path.join(self.saving_path, "checkpoint")

        # Backup old checkpoint before saving, and erase it afterwards
        to_remove = None
        if os.path.exists(checkpoint_dir):
            to_remove = os.path.join(self.saving_path, "checkpoint_old")
            shutil.move(checkpoint_dir, to_remove)

        os.mkdir(checkpoint_dir)

        # Save experiment
        # Separated function to be re-implemented by child classes to fit your
        # needs. Below is one working example.
        checkpoint_state = self._prepare_checkpoint_state()
        torch.save(checkpoint_state,
                   os.path.join(checkpoint_dir, "checkpoint_state.pkl"))

        # Save model inside the checkpoint dir
        self.model.save(checkpoint_dir)

        if to_remove:
            shutil.rmtree(to_remove)

    def _prepare_checkpoint_state(self) -> dict:
        # These are parameters that should be updated after instantiating cls.
        current_states = {
            'comet_key': self.comet_key,
            'current_epoch': self.current_epoch,
            'nb_train_batches_per_epoch': self.nb_train_batches_per_epoch,
            'nb_valid_batches_per_epoch': self.nb_valid_batches_per_epoch,
            'torch_rng_state': torch.random.get_rng_state(),
            'torch_cuda_state':
                torch.cuda.get_rng_state() if self.use_gpu else None,
            'numpy_rng_state': self.train_batch_sampler.np_rng.get_state(),
            'best_epoch_monitoring_state':
                self.best_epoch_monitoring.get_state() if
                self.best_epoch_monitoring else None,
            'train_loss_monitor_state': self.train_loss_monitor.get_state(),
            'valid_loss_monitor_state': self.valid_loss_monitor.get_state(),
            'grad_norm_monitor_state': self.grad_norm_monitor.get_state(),
            'optimizer_state': self.optimizer.state_dict(),
        }

        # Additional params are the parameters necessary to load data, batch
        # samplers/loaders (see the example script dwiml_train_model.py).
        # Note that the training set and validation set attributes should be
        # the same in theory. #toDo to be checked?
        checkpoint_state = {
            'train_sampler_params': self.train_batch_sampler.params,
            'valid_sampler_params': None,
            'train_data_params': self.train_batch_sampler.dataset.params,
            'valid_data_params': None,
            'train_loader_params': self.train_batch_loader.params,
            'valid_loader_params': None,
            'params_for_init': self.params_for_checkpoint,
            'current_states': current_states
        }

        if self.use_validation:
            checkpoint_state.update({
                'valid_sampler_params': self.valid_batch_sampler.params,
                'valid_data_params': self.valid_batch_sampler.dataset.params,
                'valid_loader_params': self.valid_batch_loader.params
            })

        return checkpoint_state

    def _update_taskman_report(self, updates):
        self.taskman_report.update(updates)
        self.taskman_report['time'] = time.time()
        self.logger.info('!taskman' + json.dumps(self.taskman_report),
                         flush=True)

    def _save_log_from_array(self, array: np.ndarray, fname: str):
        log_dir = os.path.join(self.saving_path, "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        fpath = os.path.join(log_dir, fname)
        np.save(fpath, array)

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

    @staticmethod
    def check_stopping_cause(checkpoint_state, new_patience=None,
                             new_max_epochs=None):

        # 1. Check if early stopping had been triggered.
        best_monitoring_state = \
            checkpoint_state['current_states']['best_epoch_monitoring_state']
        bad_epochs = best_monitoring_state['n_bad_epochs']
        if new_patience is None:
            # No new patience: checking if early stopping had been triggered.
            if bad_epochs >= best_monitoring_state['patience']:
                raise EarlyStoppingError(
                    "Resumed experiment was stopped because of early "
                    "stopping, increase patience in order to resume training!")
        elif bad_epochs >= new_patience:
            # New patience: checking if will be able to continue
            raise EarlyStoppingError(
                "In resumed experiment, we had reach {} bad epochs (i.e. with "
                "no improvement). You have now overriden patience to {} but "
                "that won't be enough. Please increase that value in "
                "order to resume training."
                .format(best_monitoring_state['n_bad_epochs'], new_patience))

        # 2. Checking that max_epochs had not been reached.
        current_epoch = checkpoint_state['current_states']['current_epoch']
        if new_max_epochs is None:
            if current_epoch == \
                    checkpoint_state['params_for_init']['max_epochs'] - 1:
                raise ValueError(
                    "Resumed experiment had stopped after reaching the "
                    "maximum number of epochs allowed (max_epochs = {}). "
                    "Please increase that value in order to resume training."
                    .format(checkpoint_state['params_for_init']['max_epochs']))
        else:
            if current_epoch > new_max_epochs:
                raise ValueError(
                    "In resumed experiment, we had performed {} epochs). You "
                    "have now overriden max_epoch to {} but that won't be "
                    "enough. Please increase that value in order to resume "
                    "training."
                    .format(current_epoch, new_max_epochs))


class DWIMLTrainerOneInput(DWIMLAbstractTrainer):
    def __init__(self,
                 model: MainModelAbstract, experiments_path: str,
                 experiment_name: str,
                 batch_sampler_training: DWIMLBatchSampler,
                 batch_loader_training: BatchLoaderOneInput,
                 batch_sampler_validation: DWIMLBatchSampler = None,
                 batch_loader_validation: BatchLoaderOneInput = None,
                 model_uses_streamlines: bool = False,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.01, max_epochs: int = 10,
                 max_batches_per_epoch: int = 1000, patience: int = None,
                 nb_cpu_processes: int = 0, taskman_managed: bool = False,
                 use_gpu: bool = False, comet_workspace: str = None,
                 comet_project: str = None, from_checkpoint: bool = False,
                 log_level=logging.root.level):
        super().__init__(model, experiments_path, experiment_name,
                         batch_sampler_training, batch_loader_training,
                         batch_sampler_validation, batch_loader_validation,
                         model_uses_streamlines,
                         learning_rate, weight_decay, max_epochs,
                         max_batches_per_epoch, patience,
                         nb_cpu_processes, taskman_managed, use_gpu,
                         comet_workspace, comet_project,
                         from_checkpoint, log_level)

    def run_one_batch(self, data, is_training: bool,
                      batch_loader: BatchLoaderOneInput):
        if is_training:
            # If training, enable gradients for backpropagation.
            # Uses torch's module train(), which "turns on" the training mode.
            self.model.train()
            grad_context = torch.enable_grad
        else:
            # If evaluating, turn gradients off for back-propagation
            # Uses torch's module eval(), which "turns off" the training mode.
            self.model.eval()
            grad_context = torch.no_grad

        with grad_context():
            if batch_loader.wait_for_gpu:
                if not self.use_gpu:
                    self.logger.warning(
                        "Batch sampler has been created with use_gpu=True, so "
                        "some computations have been skipped to allow user to "
                        "compute them later on GPU. Now in training, however, "
                        "you are using CPU, so this was not really useful.\n"
                        "Maybe this is an error in the code?")
                # Data interpolation has not been done yet. GPU computations
                # need to be done here in the main thread. Running final steps
                # of data preparation.
                self.logger.debug('Finalizing input data preparation on GPU.')
                batch_streamlines, final_s_ids_per_subj = data

                # Getting the inputs points from the volumes. Usually done in
                # load_batch but we preferred to wait here to have a chance to
                # run things on GPU.
                batch_inputs = batch_loader.compute_inputs(
                    batch_streamlines, final_s_ids_per_subj)

            else:
                # Data is already ready
                batch_streamlines, _, batch_inputs = data

            if is_training:
                # Reset parameter gradients
                # See here for some explanation
                # https://stackoverflow.com/questions/48001598/why-do-we-need-
                # to-call-zero-grad-in-pytorch
                self.optimizer.zero_grad()

            self.logger.debug('\n*** Computing forward propagation')
            if self.model_uses_streamlines:
                model_outputs = self.model(batch_inputs, batch_streamlines)
            else:
                model_outputs = self.model(batch_inputs)

            # Compute loss
            self.logger.debug('\n*** Computing loss')
            mean_loss = self.compute_loss(model_outputs, batch_streamlines)
            self.logger.info("Loss is : {}".format(mean_loss))

            if is_training:
                self.logger.debug('\n*** Computing back propagation')

                # Explanation on the backward here:
                # - Each parameter in the model have been created with the flag
                #   requires_grad=True by torch.
                #   ==> gradients = [i.grad for i in self.model.parameters()]
                # - When using parameters to compute something (ex, outputs)
                #   torch.autograd creates a computational graph, remembering
                #   all the functions that were used from parameters that
                #   contain the requires_grad.
                # - When calling backward, the backward of each sub-function is
                #   called iteratively, each time computing the partial
                #   derivative dloss/dw and modifying the parameters' .grad
                #   ==> model_outputs.grad_fn shows the last used function,
                #       and thus the first backward to be used, here:
                #       MeanBackward0  (last function was a mean)
                #   ==> model_outputs.grad_fn shows the last used fct.
                #       Ex, AddmmBackward  (addmm = matrix multiplication)
                mean_loss.backward()

                self.fix_parameters()

                grad_norm = compute_gradient_norm(self.model.parameters())
                self.logger.debug("Gradient norm: {}".format(grad_norm))

                # Update parameters
                self.optimizer.step()
            else:
                grad_norm = None

            if self.use_gpu:
                log_gpu_memory_usage(self.logger)

        return mean_loss.cpu().item(), grad_norm
