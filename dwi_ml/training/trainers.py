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
from dwi_ml.models.main_models import MainModelAbstract
from dwi_ml.training.batch_loaders import (
    DWIMLAbstractBatchLoader, DWIMLBatchLoaderOneInput)
from dwi_ml.training.batch_samplers import DWIMLBatchIDSampler
from dwi_ml.training.utils.gradient_norm import compute_gradient_norm
from dwi_ml.training.utils.monitoring import (
    BestEpochMonitoring, IterTimer, BatchHistoryMonitor, TimeMonitor,
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
                 learning_rates: List = None, weight_decay: float = 0.01,
                 optimizer: str = 'Adam', max_epochs: int = 10,
                 max_batches_per_epoch_training: int = 1000,
                 max_batches_per_epoch_validation: Union[int, None] = 1000,
                 patience: int = None, nb_cpu_processes: int = 0,
                 use_gpu: bool = False,
                 comet_workspace: str = None, comet_project: str = None,
                 from_checkpoint: bool = False, log_level=logging.root.level,
                 # To be deprecated
                 use_radam: bool = None, learning_rate: float = None):
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
        use_radam: bool
            Deprecated. If true, use RAdam optimizer. Else, use Adam.
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
        self.experiments_path = experiments_path
        self.experiment_name = experiment_name
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
                logging.info('Creating directory {}'.format(self.saving_path))
                os.mkdir(self.saving_path)
                os.mkdir(self.log_dir)

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
        if learning_rate is not None:
            logging.warning("Deprecated use of learning rate. Should now "
                            "be learning_rates.")
            self.learning_rates = [learning_rate]
        elif learning_rates is None:
            self.learning_rates = [0.001]
        elif isinstance(learning_rates, float):
            # Should be a list but we will accept it.
            self.learning_rates = [learning_rates]
        else:
            self.learning_rates = learning_rates
        self.weight_decay = weight_decay
        self.use_radam = use_radam

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
        self.use_gpu = use_gpu
        if self.use_radam is not None:
            logging.warning("Option --use_radam will be removed. Use option "
                            "--optimizer instead.")
            optimizer = 'RAdam' if self.use_radam else 'Adam'
        if optimizer not in ['SGD', 'Adam', 'RAdam']:
            raise ValueError("Optimizer choice {} not recognized."
                             .format(optimizer))
        self.optimizer_key = optimizer

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

        # toDo. See if we would like to implement mixed precision.
        #  Could improve performance / speed
        #  https://towardsdatascience.com/optimize-pytorch-performance-for-speed-and-memory-efficiency-2022-84f453916ea6
        #  https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/

        # ----------------------
        # Values that will be modified later on. If initializing experiment
        # from a checkpoint, these values should be updated after
        # initialization.
        # ----------------------
        if patience:
            self.best_epoch_monitoring = BestEpochMonitoring(patience)
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
        # grad_norm = The total norm (sqrt(sum(params**2))) of parameters
        # before gradient clipping, if any.
        self.train_loss_monitor = BatchHistoryMonitor(weighted=True)
        self.valid_loss_monitor = BatchHistoryMonitor(weighted=True)
        self.grad_norm_monitor = BatchHistoryMonitor(weighted=False)
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
            collate_fn=self.batch_loader.load_batch_streamlines,
            pin_memory=self.use_gpu)
        self.valid_dataloader = DataLoader(
            dataset=self.batch_sampler.dataset.validation_set,
            batch_sampler=self.batch_sampler,
            num_workers=self.nb_cpu_processes,
            collate_fn=self.batch_loader.load_batch_streamlines,
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
        self.model.set_context('training')

        # Build optimizer (Optimizer is built here since it needs the model
        # parameters)
        list_params = [n for n, _ in self.model.named_parameters()]
        self.logger.debug("Initiating trainer: {}".format(type(self)))
        self.logger.debug("This trainer will use {} optimization on the "
                          "following model.parameters: {}"
                          .format(self.optimizer_key, list_params))

        self.current_lr = self.learning_rates[0]
        if self.optimizer_key == 'RAdam':
            cls = torch.optim.RAdam
        elif self.optimizer_key == 'Adam':
            cls = torch.optim.Adam
        else:
            cls = torch.optim.SGD

        self.optimizer = cls(self.model.parameters(),
                             lr=self.current_lr, weight_decay=weight_decay)

    @property
    def params_for_checkpoint(self):
        # These are the parameters necessary to use _init_
        params = {
            'learning_rates': self.learning_rates,
            'weight_decay': self.weight_decay,
            'max_epochs': self.max_epochs,
            'max_batches_per_epoch_training': self.max_batches_per_epochs_train,
            'max_batches_per_epoch_validation': self.max_batches_per_epochs_valid,
            'nb_cpu_processes': self.nb_cpu_processes,
            'use_gpu': self.use_gpu,
            'comet_workspace': self.comet_workspace,
            'comet_project': self.comet_project,
            'optimizer': self.optimizer_key,
        }
        return params

    def save_params_to_json(self):
        """
        Utility method to save the parameters to a json file in the same
        folder as the experiment. Suggestion, call this after instantiating
        your trainer.
        """
        os.listdir(self.saving_path)
        json_filename = os.path.join(self.saving_path, "parameters.json")
        with open(json_filename, 'w') as json_file:
            json_file.write(json.dumps(
                {'Date': str(datetime.now()),
                 'Trainer params': self.params_for_checkpoint,
                 'Sampler params': self.batch_sampler.params_for_checkpoint,
                 'Loader params': self.batch_loader.params_for_checkpoint,
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
                self.comet_exp.log_parameters(self.params_for_checkpoint)
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
        self.save_params_to_json()

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
        self.optimizer.zero_grad(set_to_none=True)
        for epoch in iter_timer(range(self.current_epoch, self.max_epochs)):
            # Updating current epoch. First epoch is 0!
            self.current_epoch = epoch
            if self.comet_exp:
                self.comet_exp.set_epoch(epoch)

            self.logger.info("******* STARTING : Epoch {} (i.e. #{}) *******"
                             .format(epoch, epoch + 1))

            # Set learning rate to either current value or last value
            self.current_lr = self.learning_rates[
                min(self.current_epoch, len(self.learning_rates) - 1)]
            self.logger.info("Learning rate = {}".format(self.current_lr))
            for g in self.optimizer.param_groups:
                g['lr'] = self.current_lr

            # Training
            self.logger.info("*** TRAINING")
            self.train_one_epoch(epoch)

            # Validation
            if self.use_validation:
                self.logger.info("*** VALIDATION")
                self.validate_one_epoch(epoch)

                mean_epoch_loss = self.valid_loss_monitor.average_per_epoch[-1]
            else:
                mean_epoch_loss = self.train_loss_monitor.average_per_epoch[-1]

            # Updating info
            is_bad = self.best_epoch_monitoring.update(mean_epoch_loss, epoch)

            # Check if current best has been reached
            if self.comet_exp:
                # Saving this no matter what; we will see the stairs.
                self.comet_exp.log_metric(
                    "best_epoch",
                    self.best_epoch_monitoring.best_epoch)
            if is_bad:
                self.logger.info(
                    "** This is the {}th bad epoch (patience = {})"
                    .format(self.best_epoch_monitoring.n_bad_epochs,
                            self.best_epoch_monitoring.patience))
            elif self.best_epoch_monitoring.best_epoch == epoch:
                self._save_info_best_epoch()

            # End of epoch, save checkpoint for resuming later
            self.save_checkpoint()
            self._save_log_locally(self.train_loss_monitor.average_per_epoch,
                                   "training_loss_per_epoch.npy")
            self._save_log_locally(self.valid_loss_monitor.average_per_epoch,
                                   "validation_loss_per_epoch.npy")
            self._save_log_locally(self.grad_norm_monitor.average_per_epoch,
                                   "gradient_norm.npy")
            self._save_log_locally(self.training_time_monitor.epoch_durations,
                                   "training_epochs_duration")
            self._save_log_locally(self.validation_time_monitor.epoch_durations,
                                   "validation_epochs_duration")

            # Check for early stopping
            if self.best_epoch_monitoring.is_patience_reached:
                logger.warning(
                    "Early stopping! Loss has not improved after {} epochs!\n"
                    "Best result: {}; At epoch #{}"
                    .format(self.best_epoch_monitoring.patience,
                            self.best_epoch_monitoring.best_value,
                            self.best_epoch_monitoring.best_epoch))
                break

    def train_one_epoch(self, epoch):
        """
        Train one epoch of the model: loop on all batches (forward + backward).
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

                # Enable gradients for backpropagation.
                # Uses torch's module train(), which "turns on" the training mode.
                self.model.train()
                self.batch_loader.set_context('training')
                grad_context = torch.enable_grad
                with grad_context():
                    mean_loss, n = self.run_one_batch(data)

                self.logger.debug('*** Computing back propagation')
                mean_loss.backward()

                self.fix_parameters()  # Ex: clip gradients
                grad_norm = compute_gradient_norm(self.model.parameters())

                # Update parameters
                # toDo. We could update only every n steps.
                #  Effective batch size is n time bigger.
                #  See here https://towardsdatascience.com/optimize-pytorch-performance-for-speed-and-memory-efficiency-2022-84f453916ea6
                self.optimizer.step()

                # Reset parameter gradients to zero or to None before the next
                # forward pass
                self.optimizer.zero_grad(set_to_none=True)

                # Update information and logs
                mean_loss = mean_loss.cpu().item()
                self.train_loss_monitor.update(mean_loss, weight=n)
                self.grad_norm_monitor.update(grad_norm)
                if self.save_logs_per_batch:
                    self._update_loss_logs_after_batch(comet_context, epoch,
                                                       batch_id, mean_loss)
                    self._update_gradnorm_logs_after_batch(epoch, batch_id,
                                                           grad_norm)

            # Explicitly delete iterator to kill threads and free memory before
            # running validation
            del train_iterator

        # Saving epoch's information
        all_n = self.train_loss_monitor.current_epoch_batch_weights
        if len(all_n) == 0:
            # values not stored if loss is inf.
            logging.info("Loss was inf for all batches in this epoch. Please "
                         "supervise.")
        else:
            self.logger.info(
                "Number of data points per batch: {}\u00B1{}"
                .format(int(np.mean(all_n)), int(np.std(all_n))))
        self.train_loss_monitor.end_epoch()
        self.grad_norm_monitor.end_epoch()
        self.training_time_monitor.end_epoch()
        self._update_loss_logs_after_epoch(
            comet_context, epoch,
            self.train_loss_monitor.average_per_epoch[-1])
        self._update_gradnorm_logs_after_epoch(comet_context, epoch)

    def validate_one_epoch(self, epoch):
        """
        Validate one epoch of the model: loop on all batches.
        """
        self.validation_time_monitor.start_new_epoch()
        self.valid_loss_monitor.start_new_epoch()

        # Setting contexts
        # Turn gradients off (no back-propagation)
        # Uses torch's module eval(), which "turns off" the training mode.
        self.batch_loader.set_context('validation')
        self.batch_sampler.set_context('validation')
        comet_context = self.comet_exp.validate if self.comet_exp else None
        self.model.eval()
        self.batch_loader.set_context('validation')
        grad_context = torch.no_grad

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
                with grad_context():
                    mean_loss, n = self.run_one_batch(data)

                mean_loss = mean_loss.cpu().item()
                self.valid_loss_monitor.update(mean_loss, weight=n)
                if self.save_logs_per_batch:
                    self._update_loss_logs_after_batch(comet_context, epoch,
                                                       batch_id, mean_loss)

            # Explicitly delete iterator to kill threads and free memory before
            # running training again
            del valid_iterator

        # Save info
        self.valid_loss_monitor.end_epoch()
        self.validation_time_monitor.end_epoch()
        self._update_loss_logs_after_epoch(
            comet_context, epoch,
            self.valid_loss_monitor.average_per_epoch[-1])

    def _update_loss_logs_after_batch(self, comet_context, epoch: int,
                                      batch_id: int, mean_loss: float):
        """
        Update logs:
            - logging to user
            - save values to monitors
            - send data to comet
        """
        self.logger.info("Epoch {} (i.e. #{}): Batch loss = {}"
                         .format(epoch, epoch + 1, mean_loss))

        if self.comet_exp and batch_id % COMET_UPDATE_FREQUENCY == 0:
            with comet_context():
                # ??? epoch does not seem to show... So in fact, it only
                # saves the current epoch. Cheating and changing the metric
                # name at each epoch.
                self.comet_exp.log_metric(
                    "loss_per_batch_epoch" + str(epoch), mean_loss,
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
        self.logger.info("   Mean loss for this epoch: {}".format(loss))

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
        if self.comet_exp:
            with comet_context():
                self.comet_exp.log_metric(
                    "mean_gradient_norm_per_epoch",
                    self.grad_norm_monitor.average_per_epoch[epoch],
                    epoch=epoch, step=None)

    def _save_info_best_epoch(self):
        self.logger.info("   Best epoch yet! Saving model and loss history.")

        # Save model
        self.model.save_params_and_state(
            os.path.join(self.saving_path, 'best_model'))

        best_losses = {
            'train_loss':
                self.train_loss_monitor.average_per_epoch[
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
              streamlines)

        Returns
        -------
        mean_loss : float
            The mean loss of the provided batch.
        n: int
            Total number of points for this batch.
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
        trainer = cls(model=model,
                      experiments_path=experiments_path,
                      experiment_name=experiment_name,
                      batch_sampler=batch_sampler, batch_loader=batch_loader,
                      from_checkpoint=True, log_level=log_level,
                      **checkpoint_state['params_for_init'])

        current_states = checkpoint_state['current_states']
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
        if new_patience:
            trainer.best_epoch_monitoring.patience = new_patience
        logging.info("Starting from checkpoint! Starting from epoch #{}.\n"
                     "Previous best model was discovered at epoch #{}.\n"
                     "When finishing previously, we were counting {} epochs "
                     "with no improvement."
                     .format(trainer.current_epoch,
                             trainer.best_epoch_monitoring.best_epoch,
                             trainer.best_epoch_monitoring.n_bad_epochs))

        return trainer

    def save_checkpoint(self):
        """
        Save an experiment checkpoint that can be resumed from.
        """
        self.logger.debug("Saving checkpoint...")

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
            'batch_sampler_params': self.batch_sampler.params_for_checkpoint,
            'batch_loader_params': self.batch_loader.params_for_checkpoint,
            'params_for_init': self.params_for_checkpoint,
            'current_states': current_states
        }

        return checkpoint_info

    def _save_log_locally(self, array: np.ndarray, fname: str):
        np.save(os.path.join(self.log_dir, fname), array)

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

        current_epoch = checkpoint_state['current_states']['current_epoch']

        # 1. Check if early stopping had been triggered.
        best_monitoring_state = \
            checkpoint_state['current_states']['best_epoch_monitoring_state']
        bad_epochs = best_monitoring_state['n_bad_epochs']
        if new_patience is None:
            # No new patience: checking if early stopping had been triggered.
            if bad_epochs >= best_monitoring_state['patience']:
                raise EarlyStoppingError(
                    "Resumed experiment was stopped because of early "
                    "stopping (patience {} reached at epcoh {}).\n"
                    "Increase patience in order to resume training!"
                    .format(best_monitoring_state['patience'],
                            current_epoch))
        elif bad_epochs >= new_patience:
            # New patience: checking if will be able to continue
            raise EarlyStoppingError(
                "In resumed experiment, we had reach {} bad epochs (i.e. with "
                "no improvement). You have now overriden patience to {} but "
                "that won't be enough. Please increase that value in "
                "order to resume training."
                .format(best_monitoring_state['n_bad_epochs'], new_patience))

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
                    "training."
                    .format(current_epoch + 1, new_max_epochs))


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
              streamlines)

        Returns
        -------
        mean_loss : float
            The mean loss of the provided batch.
        n: int
            Total number of points for this batch.
        """
        # Data interpolation has not been done yet. GPU computations
        # need to be done here in the main thread. Running final steps
        # of data preparation.
        self.logger.debug('Finalizing input data preparation on GPU.')
        streamlines, final_s_ids_per_subj = data

        # Dataloader always works on CPU.
        # We have the possibility to bring everything to GPU now.
        # Sending to right device. Model should already be moved.
        streamlines = [s.to(self.device, non_blocking=True, dtype=torch.float)
                       for s in streamlines]

        # Getting the inputs points from the volumes.
        # Uses the model's method, with the batch_loader's data.
        streamlines_f = self.model.prepare_streamlines_f(streamlines)
        batch_inputs = self.batch_loader.load_batch_inputs(
            streamlines_f, final_s_ids_per_subj)

        # Possibly add noise to inputs here.
        self.logger.debug('*** Computing forward propagation')
        if self.model.forward_uses_streamlines:
            # Now possibly add noise to streamlines (training / valid)
            streamlines_f = self.batch_loader.add_noise_streamlines(
                streamlines_f, self.device)

            logger.debug("Uses a batch of {} streamlines, {} coordinates, "
                         "{} input points."
                         .format(len(streamlines),
                                 sum([len(s) for s in streamlines_f]),
                                 sum([len(s) for s in batch_inputs])))

            # Possibly computing directions twice (during forward and loss)
            # but ok, shouldn't be too heavy. Easier to deal with multiple
            # project's requirements by sending whole streamlines rather
            # than only directions.
            model_outputs = self.model(batch_inputs, streamlines_f)
            del streamlines_f
        else:
            logger.debug("Uses a batch of {} streamlines, {} input points."
                         .format(len(streamlines),
                                 sum([len(s) for s in batch_inputs])))
            model_outputs = self.model(batch_inputs)

        self.logger.debug('*** Computing loss')
        logger.debug("Uses a batch of {} streamlines, {} coordinates, "
                     "{} outputs."
                     .format(len(streamlines),
                             sum([len(s) for s in streamlines]),
                             len(model_outputs)))
        if self.model.loss_uses_streamlines:
            mean_loss, n = self.model.compute_loss(model_outputs, streamlines)
        else:
            mean_loss, n = self.model.compute_loss(model_outputs)

        if self.use_gpu:
            log_gpu_memory_usage(self.logger)

        # The mean tensor is a single value. Converting to float using item().
        return mean_loss, n
