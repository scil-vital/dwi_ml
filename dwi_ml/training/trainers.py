# -*- coding: utf-8 -*-
from datetime import datetime
import json
import logging
import os
import shutil
from typing import Union, List

import numpy as np
import torch
from comet_ml import (Experiment as CometExperiment, ExistingExperiment)
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from dwi_ml.data.processing.streamlines.post_processing import \
    compute_directions
from dwi_ml.experiment_utils.memory import log_gpu_memory_usage
from dwi_ml.experiment_utils.tqdm_logging import tqdm_logging_redirect
from dwi_ml.models.main_models import MainModelAbstract, ModelForTracking
from dwi_ml.training.batch_loaders import (
    DWIMLAbstractBatchLoader, DWIMLBatchLoaderOneInput)
from dwi_ml.training.batch_samplers import DWIMLBatchIDSampler
from dwi_ml.training.gradient_norm import compute_gradient_norm
from dwi_ml.training.monitoring import (
    BestEpochMonitoring, IterTimer, ValueHistoryMonitor, TimeMonitor,
    EarlyStoppingError)

logger = logging.getLogger('train_logger')
# If the remaining time is less than one epoch + X seconds, we will quit
# training now, to allow saving time.
QUIT_TIME_DELAY_SECONDS = 10
# Update comet every 10 batch for faster management.
COMET_UPDATE_FREQUENCY = 10


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

    NOTE: TRAINER USES STREAMLINES COORDINATES IN VOXEL SPACE, TO CORNER.
    """
    # For now, this is ugly... But the option is there if you want.
    save_logs_per_batch = False

    def __init__(self,
                 model: MainModelAbstract, experiments_path: str,
                 experiment_name: str,
                 batch_sampler: DWIMLBatchIDSampler,
                 batch_loader: DWIMLAbstractBatchLoader,
                 learning_rate: float = 0.001, weight_decay: float = 0.01,
                 use_radam: bool = False, betas: List[float] = None,
                 max_epochs: int = 10,
                 max_batches_per_epoch_training: int = 1000,
                 max_batches_per_epoch_validation: Union[int, None] = 1000,
                 patience: int = None, nb_cpu_processes: int = 0,
                 use_gpu: bool = False,
                 comet_workspace: str = None, comet_project: str = None,
                 from_checkpoint: bool = False,
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
        batch_sampler: DWIMLBatchIDSampler
            Instantiated class used for sampling batches.
            Data in batch_sampler.dataset must be already loaded.
        batch_loader: DWIMLAbstractBatchLoader
            Instantiated class with a load_batch method able to load data
            associated to sampled batch ids. Data in batch_sampler.dataset must
            be already loaded.
        learning_rate: float
            Learning rate. Default: 0.001 (torch's default)
        weight_decay: float
            Add a weight decay penalty on the parameters. Default: 0.01.
            (torch's default).
        use_radam: bool
            If true, use RAdam optimizer. Else, use Adam.
        betas: List[float]
            With RAdam optimizer, beta values. Default: None (use torch's
            default; (0.9, 0.999).
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
        nb_cpu_processes: int
            Number of parallel CPU workers. Use 0 to avoid parallel threads.
            Default : 0.
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
        self.experiment_name = experiment_name
        self.saving_path = os.path.join(experiments_path,
                                        experiment_name)
        if not from_checkpoint and not os.path.isdir(self.saving_path):
            logging.info('Creating directory {}'.format(self.saving_path))
            os.mkdir(self.saving_path)

        # Note that the training/validation sets are also contained in the
        # data_loaders.dataset
        self.batch_sampler = batch_sampler
        if self.batch_sampler.dataset.validation_set.nb_subjects == 0:
            self.use_validation = False
            self.logger.warning(
                "WARNING! There is no validation set. Loss for best epoch "
                "monitoring will be the training loss. \n"
                "Best practice is to have a validation set.")
        else:
            self.use_validation = True
        self.batch_loader = batch_loader
        self.model = model
        self.max_epochs = max_epochs
        self.max_batches_per_epochs_train = max_batches_per_epoch_training
        self.max_batches_per_epochs_valid = max_batches_per_epoch_validation
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.use_radam = use_radam
        if self.betas is not None and self.use_radam:
            logger.warning("Beta values were given, but use_radam option was "
                           "not chosen. This value is discarded.")
            self.betas = None
        if self.betas is None and self.use_radam:
            self.betas = (0.9, 0.99)
        self.patience = patience
        self.nb_cpu_processes = nb_cpu_processes
        self.use_gpu = use_gpu

        self.comet_workspace = comet_workspace
        self.comet_project = comet_project

        # ----------------------
        # Values fixed by us
        # ----------------------
        self.space = 'vox'
        self.origin = 'corner'

        # Device and rng value. Note that if loading from a checkpoint, the
        # complete state should be updated.
        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')

                # If you see a hint error below in code editor, upgrade torch.
                torch.cuda.manual_seed(self.batch_sampler.rng)

                logging.info("We will be using GPU!")
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
                patience=self.max_epochs + 1)

        self.current_epoch = 0

        # Nb of batches with be estimated later on
        # This will be used to estimate time left for nice prints to user.
        self.nb_train_batches_per_epoch = None
        self.nb_valid_batches_per_epoch = None

        # RNG state
        # Nothing to do here.

        # Setup monitors
        self.train_loss_monitor = ValueHistoryMonitor("Training loss")
        self.valid_loss_monitor = ValueHistoryMonitor("Validation loss")
        self.grad_norm_monitor = ValueHistoryMonitor("Grad Norm")
        self.training_time_monitor = TimeMonitor()
        self.validation_time_monitor = TimeMonitor()

        # Comet values will be instantiated in train().
        self.comet_exp = None
        self.comet_key = None

        # ---------------------
        # Dataloader: usage will depend on context
        # ---------------------
        # Create DataLoaders from the BatchSamplers
        #   * Before usage, context must be set for the batch sampler and the
        #     batch loader, to use appropriate parameters.
        #   * Pin memory if interpolation is done by workers; this means that
        #     dataloader output is on GPU, ready to be fed to the model.
        #     Otherwise, dataloader output is kept on CPU, and the main thread
        #     sends volumes and coords on GPU for interpolation.
        self.logger.debug("- Instantiating dataloaders...")
        self.train_dataloader = DataLoader(
            dataset=self.batch_sampler.dataset.training_set,
            batch_sampler=self.batch_sampler,
            num_workers=self.nb_cpu_processes,
            collate_fn=self.batch_loader.load_batch,
            pin_memory=self.use_gpu)
        self.valid_dataloader = DataLoader(
            dataset=self.batch_sampler.dataset.validation_set,
            batch_sampler=self.batch_sampler,
            num_workers=self.nb_cpu_processes,
            collate_fn=self.batch_loader.load_batch,
            pin_memory=self.use_gpu)

        # ----------------------
        # Launching optimizer!
        # ----------------------

        # Prepare optimizer
        # Send model to device. Reminder, contrary to tensors, model.to
        # overwrites the model.
        # NOTE: This ordering is important! The optimizer needs to use the cuda
        # Tensors if using the GPU...
        self.model.move_to(device=self.device)

        # Build optimizer (Optimizer is built here since it needs the model
        # parameters)
        list_params = [n for n, _ in self.model.named_parameters()]
        self.logger.debug("Initiating trainer: {}".format(type(self)))
        self.logger.debug("This trainer will use Adam optimization on the "
                          "following model.parameters: {}".format(list_params))

        if self.use_radam:
            self.optimizer = torch.optim.RAdam(self.model.parameters(),
                                               lr=learning_rate,
                                               betas=self.betas,
                                               weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=learning_rate,
                                              weight_decay=weight_decay)

    @property
    def params_for_checkpoint(self):
        # These are the parameters necessary to use _init_
        params = {
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'max_epochs': self.max_epochs,
            'max_batches_per_epoch_training': self.max_batches_per_epochs_train,
            'max_batches_per_epoch_validation': self.max_batches_per_epochs_valid,
            'patience': self.patience,
            'nb_cpu_processes': self.nb_cpu_processes,
            'use_gpu': self.use_gpu,
            'comet_workspace': self.comet_workspace,
            'comet_project': self.comet_project,
        }
        return params

    @property
    def params_for_json_prints(self) -> dict:
        params = self.params_for_checkpoint
        params.update({
            'experiments_path': self.experiments_path,
            'experiment_name': self.experiment_name,
            'type': str(type(self)),
            'comet_key': self.comet_key,
            'computed_values': {
                'nb_training_batches_per_epoch':
                    self.nb_train_batches_per_epoch,
                'nb_validation_batches_per_epoch':
                    self.nb_valid_batches_per_epoch
            }
        })
        return params

    def save_params_to_json(self):
        """
        Utility method to save the parameters to a json file in the same
        folder as the experiment. Suggestion, call this after instantiating
        your trainer.
        """
        json_filename = os.path.join(self.saving_path, "parameters.json")
        with open(json_filename, 'w') as json_file:
            json_file.write(json.dumps(
                {'Date': str(datetime.now()),
                 'Trainer params': self.params_for_json_prints,
                 'Sampler params': self.batch_sampler.params,
                 'Batch loader params':
                     self.batch_loader.params_for_json_prints,
                 },
                indent=4, separators=(',', ': ')))

    def _init_comet(self):
        """
        For more information on comet, see our doc/Getting Started
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
                self.comet_exp.log_parameters(self.params_for_json_prints)
                self.comet_key = self.comet_exp.get_key()
            elif self.comet_project:
                raise ValueError("You have provided a comet project, "
                                 "but no comet workspace!")
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
        streamline_group = self.batch_sampler.streamline_group_idx
        train_set = self.batch_sampler.dataset.training_set
        valid_set = self.batch_sampler.dataset.validation_set

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

        if self.use_validation:  # Verifying or else, could divide by 0.
            nb_valid /= self.batch_sampler.batch_size_validation
            final_nb_valid = min(nb_valid, self.max_batches_per_epochs_valid)
        else:
            final_nb_valid = 0

        return int(final_nb_train), int(final_nb_valid)

    def train_and_validate(self):
        """
        Train + validates the model (+ computes loss)

        - Starts comet,
        - Creates DataLoaders from the BatchSamplers,
        - For each epoch
            - uses _train_one_epoch and _validate_one_epoch,
            - checks for earlyStopping if the loss is bad,
            - saves the model if the loss is good.
        - Checks if allowed training time is exceeded.

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

        # Instantiating comet
        self._init_comet()

        # Instantiating our IterTimer.
        # After each iteration, checks that the maximum allowed time has not
        # been reached.
        iter_timer = IterTimer(history_len=20)

        # Start from current_epoch in case the experiment is resuming
        # Train each epoch
        for epoch in iter_timer(range(self.current_epoch, self.max_epochs)):
            # Updating current epoch. First epoch is 0!
            self.current_epoch = epoch
            if self.comet_exp:
                self.comet_exp.set_epoch(epoch)

            # Training
            self.logger.info("**********TRAINING: Epoch #{}*************"
                             .format(epoch + 1))
            self.train_one_epoch(epoch)

            # Validation
            if self.use_validation:
                self.logger.info("**********VALIDATION: Epoch #{}*************"
                                 .format(epoch + 1))
                self.validate_one_epoch(epoch)

                mean_epoch_loss = self.valid_loss_monitor.epochs_means[epoch]
            else:
                mean_epoch_loss = self.train_loss_monitor.epochs_means[epoch]

            # Updating info
            self.best_epoch_monitoring.update(mean_epoch_loss, epoch)

            # Check if current best has been reached
            if self.best_epoch_monitoring.best_epoch == epoch:
                self._save_info_best_epoch()

            # End of epoch, save checkpoint for resuming later
            self.save_checkpoint()

            # Check for early stopping
            if self.best_epoch_monitoring.is_patience_reached:
                logger.warning(
                    "Early stopping! Loss has not improved after {} epochs!\n"
                    "Best result: {}; At epoch #{}"
                    .format(self.patience,
                            self.best_epoch_monitoring.best_value,
                            self.best_epoch_monitoring.best_epoch))
                break

        self._save_log_from_array(self.train_loss_monitor.epochs_means,
                                  "training_loss_per_epoch.npy")
        self._save_log_from_array(self.valid_loss_monitor.epochs_means,
                                  "validation_loss_per_epoch.npy")
        self._save_log_from_array(self.grad_norm_monitor.epochs_means,
                                  "gradient_norm.npy")
        self._save_log_from_array(self.training_time_monitor.epoch_durations,
                                  "training_epochs_duration")
        self._save_log_from_array(self.validation_time_monitor.epoch_durations,
                                  "validation_epochs_duration")

    def train_one_epoch(self, epoch):
        """
        Train one epoch of the model: loop on all batches.

        All *args will be passed all to run_one_batch, which you should
        implement, in case you need some variables.
        """
        self.training_time_monitor.start_new_epoch()
        self.train_loss_monitor.start_new_epoch()
        self.grad_norm_monitor.start_new_epoch()

        # Setting contexts
        self.batch_loader.set_context('training')
        self.batch_sampler.set_context('training')
        comet_context = self.comet_exp.train if self.comet_exp else None

        # Make sure there are no existing HDF handles if using parallel workers
        if (self.nb_cpu_processes > 0 and
                self.batch_sampler.context_subset.is_lazy):
            self.batch_sampler.context_subset.close_all_handles()
            self.batch_sampler.context_subset.volume_cache_manager = None

        if self.comet_exp:
            self.comet_exp.log_metric("current_epoch", self.current_epoch)

        # Training all batches
        self.logger.debug("Training one epoch: iterating on batches using "
                          "tqdm on the dataloader...")

        # Note: loggers = [logging.root] only.
        # If we add our sub-loggers there, they duplicate.
        # A handler is added in the root logger, and sub-loggers propagate
        # their message.
        with tqdm_logging_redirect(self.train_dataloader, ncols=100,
                                   total=self.nb_train_batches_per_epoch,
                                   loggers=[logging.root],
                                   tqdm_class=tqdm) as pbar:

            train_iterator = enumerate(pbar)
            for batch_id, data in train_iterator:
                # Break if maximum number of epochs has been reached
                if batch_id == self.nb_train_batches_per_epoch:
                    # Explicitly close tqdm's progress bar to fix possible bugs
                    # when breaking the loop
                    pbar.close()
                    break

                batch_mean_loss, grad_norm = self.run_one_batch(
                    data, is_training=True)

                # Update information and logs
                self.train_loss_monitor.update(batch_mean_loss)
                self.grad_norm_monitor.update(grad_norm)
                if self.save_logs_per_batch:
                    self._update_loss_logs_after_batch(
                        comet_context, epoch, batch_id, batch_mean_loss)
                    self._update_gradnorm_logs_after_batch(epoch, batch_id,
                                                           grad_norm)

            # Explicitly delete iterator to kill threads and free memory before
            # running validation
            del train_iterator

        # Saving epoch's information
        self.logger.info("Finishing epoch...")
        self.train_loss_monitor.end_epoch()
        self.grad_norm_monitor.end_epoch()
        self.training_time_monitor.end_epoch()
        self._update_loss_logs_after_epoch(
            comet_context, epoch, self.train_loss_monitor.epochs_means[epoch])
        self._update_gradnorm_logs_after_epoch(comet_context, epoch)

    def validate_one_epoch(self, epoch):
        """
        Validate one epoch of the model: loop on all batches.

        All *args will be passed all to run_one_batch, which you should
        implement, in case you need some variables.
        """
        self.validation_time_monitor.start_new_epoch()
        self.valid_loss_monitor.start_new_epoch()

        # Setting contexts
        self.batch_loader.set_context('validation')
        self.batch_sampler.set_context('validation')
        comet_context = self.comet_exp.validate if self.comet_exp else None

        # Make sure there are no existing HDF handles if using parallel workers
        if (self.nb_cpu_processes > 0 and
                self.batch_sampler.context_subset.is_lazy):
            self.batch_sampler.context_subset.close_all_handles()
            self.batch_sampler.context_subset.volume_cache_manager = None

        # Validate all batches
        with tqdm_logging_redirect(self.valid_dataloader, ncols=100,
                                   total=self.nb_valid_batches_per_epoch,
                                   loggers=[logging.root],
                                   tqdm_class=tqdm) as pbar:
            valid_iterator = enumerate(pbar)
            for batch_id, data in valid_iterator:
                # Break if maximum number of epochs has been reached
                if batch_id == self.nb_valid_batches_per_epoch:
                    # Explicitly close tqdm's progress bar to fix possible bugs
                    # when breaking the loop
                    pbar.close()
                    break

                # Validate this batch: forward propagation + loss
                batch_mean_loss, _ = self.run_one_batch(
                    data, is_training=False)

                self.valid_loss_monitor.update(batch_mean_loss)
                if self.save_logs_per_batch:
                    self._update_loss_logs_after_batch(
                        comet_context, epoch, batch_id, batch_mean_loss)

            # Explicitly delete iterator to kill threads and free memory before
            # running training again
            del valid_iterator

        # Save info
        self.valid_loss_monitor.end_epoch()
        self.validation_time_monitor.end_epoch()
        self._update_loss_logs_after_epoch(
            comet_context, epoch, self.valid_loss_monitor.epochs_means[epoch])

    def _update_loss_logs_after_batch(
            self, comet_context, epoch: int, batch_id: int,
            batch_mean_loss: float):
        """
        Update logs:
            - logging to user
            - save values to monitors
            - send data to comet
        """
        self.logger.info("Epoch {}: Batch loss = {}"
                         .format(epoch + 1, batch_mean_loss))

        if self.comet_exp and batch_id % COMET_UPDATE_FREQUENCY == 0:
            with comet_context():
                # ??? epoch does not seem to show... So in fact, it only
                # saves the current epoch. Cheating and changing the metric
                # name at each epoch.
                self.comet_exp.log_metric(
                    "loss_per_batch_epoch" + str(epoch), batch_mean_loss,
                    step=batch_id, epoch=0)

    def _update_gradnorm_logs_after_batch(self, epoch: int, batch_id: int,
                                          grad_norm: float):
        """
        Update logs:
            - save values to monitors
            - send data to comet

        Should only be done during training
        """
        if self.comet_exp and batch_id % COMET_UPDATE_FREQUENCY == 0:
            self.comet_exp.log_metric(
                "gradient_norm_per_batch_epoch" + str(epoch),
                grad_norm, step=batch_id, epoch=epoch)

    def _update_loss_logs_after_epoch(self, comet_context, epoch: int,
                                      loss: float):
        """
        Update logs:
            - logging to user
            - get values from monitors and save final log locally.
            - send mean data to comet

        local_context: prefix when saving log. Training_ or Validate_ for
        instance.
        """
        self.logger.info("Mean loss for this epoch: {}".format(loss))

        if self.comet_exp:
            with comet_context():
                # Not really implemented yet.
                # See https://github.com/comet-ml/issue-tracking/issues/247
                # Cheating. To have a correct plotting per epoch (no step)
                # using step = epoch. In comet_ml, it is intended to be
                # step = batch.
                self.comet_exp.log_metric("loss_per_epoch", loss, epoch=0,
                                          step=epoch)

    def _update_gradnorm_logs_after_epoch(self, comet_context, epoch: int):
        self.logger.info(
            "Mean gradient norm : {}"
            .format(self.grad_norm_monitor.epochs_means[epoch]))

        if self.comet_exp:
            with comet_context():
                self.comet_exp.log_metric(
                    "mean_gradient_norm_per_epoch",
                    self.grad_norm_monitor.epochs_means[epoch],
                    epoch=epoch, step=None)

    def _save_info_best_epoch(self):
        self.logger.info("Best epoch yet! Saving model and loss history.")

        # Save model
        self.model.save_params_and_state(
            os.path.join(self.saving_path, 'best_model'))

        best_losses = {
            'train_loss':
                self.train_loss_monitor.epochs_means[
                    self.best_epoch_monitoring.best_epoch],
            'valid_loss':
                self.best_epoch_monitoring.best_value if
                self.use_validation else None}
        with open(os.path.join(self.saving_path, "best_epoch_losses.json"),
                  'w') as json_file:
            json_file.write(json.dumps(best_losses, indent=4,
                                       separators=(',', ': ')))

        if self.comet_exp:
            self.comet_exp.log_metric(
                "best_loss",
                self.best_epoch_monitoring.best_value)
            self.comet_exp.log_metric(
                "best_epoch",
                self.best_epoch_monitoring.best_epoch)

    def run_one_batch(self, data, is_training: bool):
        """
        Run a batch of data through the model (calling its forward method)
        and return the mean loss. If training, run the backward method too.

        Parameters
        ----------
        data : tuple of (List, dict)
            This is the output of the AbstractBatchLoader's load_batch()
            method. If wait_for_gpu, data is
            (batch_streamlines, final_streamline_ids_per_subj). Else, data is
            (batch_streamlines, final_streamline_ids_per_subj, inputs)
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
            batch_sampler: DWIMLBatchIDSampler,
            batch_loader: DWIMLAbstractBatchLoader,
            checkpoint_state: dict, new_patience,
            new_max_epochs, log_level):
        """
        During save_checkpoint(), checkpoint_state.pkl is saved. Loading it
        back offers a dict that can be used to instantiate an experiment and
        set it at the same state as previously. (Current_epoch is updated +1).

        Hint: If you want to use this in your child class, use:
        experiment, checkpoint_state = super(cls, cls).init_from_checkpoint(...
        """
        print(cls)
        trainer = cls(model=model,
                      experiments_path=experiments_path,
                      experiment_name=experiment_name,
                      batch_sampler=batch_sampler, batch_loader=batch_loader,
                      from_checkpoint=True, log_level=log_level,
                      **checkpoint_state['params_for_init'])

        current_states = checkpoint_state['current_states']

        # Overriding values
        if new_patience:
            trainer.patience = new_patience
        if new_max_epochs:
            trainer.max_epochs = new_max_epochs

        # Save params to json to help user remember.
        now = datetime.now()
        filename = os.path.join(trainer.saving_path,
                                "parameters_{}.json"
                                .format(now.strftime("%d-%m-%Y_%Hh%M")))
        with open(filename, 'w') as json_file:
            json_file.write(json.dumps(
                {'Date': str(datetime.now()),
                 'New patience': new_patience,
                 'New max_epochs': new_max_epochs
                 },
                indent=4, separators=(',', ': ')))

        # Set RNG states
        torch.set_rng_state(current_states['torch_rng_state'])
        trainer.batch_sampler.np_rng.set_state(
            current_states['numpy_rng_state'])
        if trainer.use_gpu:
            torch.cuda.set_rng_state(current_states['torch_cuda_state'])

        # Comet properties
        trainer.comet_key = current_states['comet_key']

        # Starting one epoch further than last time!
        trainer.current_epoch = current_states['current_epoch'] + 1

        # Computed values
        trainer.nb_train_batches_per_epoch = \
            current_states['nb_train_batches_per_epoch']
        trainer.nb_valid_batches_per_epoch = \
            current_states['nb_valid_batches_per_epoch']

        # Monitors
        trainer.best_epoch_monitoring.set_state(
            current_states['best_epoch_monitoring_state'])
        trainer.train_loss_monitor.set_state(
            current_states['train_loss_monitor_state'])
        trainer.valid_loss_monitor.set_state(
            current_states['valid_loss_monitor_state'])
        trainer.grad_norm_monitor.set_state(
            current_states['grad_norm_monitor_state'])
        trainer.optimizer.load_state_dict(current_states['optimizer_state'])

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
        checkpoint_state = self._prepare_checkpoint_info()
        torch.save(checkpoint_state,
                   os.path.join(checkpoint_dir, "checkpoint_state.pkl"))

        # Save model inside the checkpoint dir
        self.model.save_params_and_state(os.path.join(checkpoint_dir, 'model'))

        if to_remove:
            shutil.rmtree(to_remove)

    def _prepare_checkpoint_info(self) -> dict:
        # These are parameters that should be updated after instantiating cls.
        current_states = {
            'comet_key': self.comet_key,
            'current_epoch': self.current_epoch,
            'nb_train_batches_per_epoch': self.nb_train_batches_per_epoch,
            'nb_valid_batches_per_epoch': self.nb_valid_batches_per_epoch,
            'torch_rng_state': torch.random.get_rng_state(),
            'torch_cuda_state':
                torch.cuda.get_rng_state() if self.use_gpu else None,
            'numpy_rng_state': self.batch_sampler.np_rng.get_state(),
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
        checkpoint_info = {
            # todo Verify:
            #  batch sampler and batch loader should have the same dataset
            'dataset_params': self.batch_sampler.dataset.params,
            'batch_sampler_params': self.batch_sampler.params,
            'batch_loader_params': self.batch_loader.params_for_checkpoint,
            'params_for_init': self.params_for_checkpoint,
            'current_states': current_states
        }

        return checkpoint_info

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
                    "maximum number of epochs allowed (max_epochs = {}).\n"
                    "Please increase that value in order to resume training."
                    .format(checkpoint_state['params_for_init']['max_epochs']))
        else:
            if current_epoch + 1 >= new_max_epochs:
                raise ValueError(
                    "In resumed experiment, we had performed {} epochs. \nYou "
                    "have now overriden max_epoch to {} but that won't be "
                    "enough. \nPlease increase that value in order to resume "
                    "training."
                    .format(current_epoch + 1, new_max_epochs))


class DWIMLTrainerOneInput(DWIMLAbstractTrainer):
    batch_loader: DWIMLBatchLoaderOneInput

    def __init__(self, save_estimated_outputs: bool = False, **kw):
        """
        Params
        ------
        save_estimated_outputs: bool
            If true, during validation (or training, if there is no validation
            set), and if model allows it, we will send additional parameters
            to the forward method to save outputs to allow visual supervision
            of the model.
        """
        super().__init__(**kw)
        self.save_estimated_outputs = save_estimated_outputs

    def run_one_batch(self, data, is_training: bool):
        """
        Run a batch of data through the model (calling its forward method)
        and return the mean loss. If training, run the backward method too.

        If the batch_loader was instantiated with wait_for_gpu, then we need to
        compute the inputs here; not done yet.

        Parameters
        ----------
        data : tuple of (List, dict)
            This is the output of the AbstractBatchLoader's load_batch()
            method. If wait_for_gpu, data is
            (batch_streamlines, final_streamline_ids_per_subj). Else, data is
            (batch_streamlines, final_streamline_ids_per_subj, inputs)
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
        save_estimated_outputs = self.save_estimated_outputs
        if is_training:
            # If training, enable gradients for backpropagation.
            # Uses torch's module train(), which "turns on" the training mode.
            self.model.train()
            self.batch_loader.set_context('training')
            grad_context = torch.enable_grad
            if self.use_validation:
                save_estimated_outputs = False
        else:
            # If evaluating, turn gradients off for back-propagation
            # Uses torch's module eval(), which "turns off" the training mode.
            self.model.eval()
            self.batch_loader.set_context('validation')
            grad_context = torch.no_grad

        with grad_context():
            if self.batch_loader.wait_for_gpu:
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

                # toDo. Could we convert streamlines to tensor already when
                #  loading, and send to GPU here? Would need to modify methods
                #  such as extend_neighborhood_with_coords.

                # Getting the inputs points from the volumes. Usually done in
                # load_batch but we preferred to wait here to have a chance to
                # run things on GPU.
                batch_inputs = self.batch_loader.compute_inputs(
                    batch_streamlines, final_s_ids_per_subj,
                    device=self.device)

            else:
                # Data is already ready
                batch_streamlines, final_s_ids_per_subj, batch_inputs = data

            lengths = [len(s) - 1 for s in batch_streamlines]
            logger.debug("Loaded a batch of {} streamlines, {} inputs points"
                         .format(len(batch_streamlines), sum(lengths)))
            logger.debug("Loaded the associated {} inputs.".
                         format(len(batch_inputs)))
            if is_training:
                # Reset parameter gradients
                # See here for some explanation
                # https://stackoverflow.com/questions/48001598/why-do-we-need-
                # to-call-zero-grad-in-pytorch
                self.optimizer.zero_grad()

            forward_kwargs = {}
            loss_kwargs = {}
            if self.model.model_uses_streamlines:
                logger.debug("Sending streamlines to tensor and using in "
                             "model.")
                batch_streamlines = [torch.tensor(s).to(self.device) for s in
                                     batch_streamlines]
                forward_kwargs.update({'target_streamlines': batch_streamlines})
                loss_kwargs.update({'target_streamlines': batch_streamlines})
            if self.model.model_uses_dirs:
                logger.debug("Computing directions and using in model.")
                dirs = compute_directions(batch_streamlines, self.device)
                forward_kwargs.update({'target_dirs': dirs})
                loss_kwargs.update({'target_dirs': dirs})

                if (save_estimated_outputs and
                        isinstance(self.model, ModelForTracking) and
                        self.model.allow_saving_estimated_outputs):
                    logger.debug("Getting reference; we wil save estimated "
                                 "outputs as a sft.")
                    # If we allow it, it means any subject in the batch sampler
                    # can be used as ref.
                    # ids is a Dict[int, slice]
                    ref_subj = final_s_ids_per_subj.keys()[0]

                    # Getting the subject (no need to load it here, just
                    # getting its affine, always loaded even with lazy).
                    ref = self.batch_loader.context_subset.get_volume(
                        ref_subj, self.batch_loader.input_group_idx,
                        load_it=False).affine
                    path = os.path.join(self.saving_path,
                                        'latest_batch_outputs')

                    forward_kwargs.update({'ref': ref,
                                           'saving_path': path,
                                           'space': self.space,
                                           'origin': self.origin})

            self.logger.debug('*** Computing forward propagation')
            model_outputs = self.model(batch_inputs, **forward_kwargs)

            # Compute loss
            self.logger.debug('*** Computing loss')
            mean_loss = self.model.compute_loss(model_outputs, **loss_kwargs)

            if is_training:
                self.logger.debug('*** Computing back propagation')

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

                # Update parameters
                self.optimizer.step()
            else:
                grad_norm = None

            if self.use_gpu:
                log_gpu_memory_usage(self.logger)

        return mean_loss.cpu().item(), grad_norm
