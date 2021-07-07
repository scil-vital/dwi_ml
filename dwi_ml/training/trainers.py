# -*- coding: utf-8 -*-
import json
import logging
import os
import shutil
import time

from comet_ml import (Experiment as CometExperiment, ExistingExperiment)
import contextlib2
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from dwi_ml.experiment.monitoring import (
    EarlyStopping, EarlyStoppingError, IterTimer, ValueHistoryMonitor)
from dwi_ml.model.batch_samplers import BatchSamplerAbstract
from dwi_ml.model.main_models import ModelAbstract

# If the remaining time is less than one epoch + X seconds, we will quit
# training now, to allow updating taskman_report.
QUIT_TIME_DELAY_SECONDS = 30
# This trainer's version
VERSION = 0


class DWIMLTrainer:
    """
    This Trainer class's train() method:
        - Creates DataLoaders from the batch_samplers. Collate_fn will be the
        sampler.load_batch() method, and the dataset will be
        sampler.source_data.
        - Trains each epoch by using compute_batch_loss, which should be
        implemented in each project's child class.

    Comet is used to save training information, but some logs will also be
    saved locally in the experiment_path.
    """
    def __init__(self,
                 batch_sampler_training: BatchSamplerAbstract,
                 batch_sampler_validation: BatchSamplerAbstract,
                 model: ModelAbstract,
                 experiment_path: str, experiment_name: str,
                 learning_rate: float = None, weight_decay: float = None,
                 max_epochs: int = None, max_batches_per_epoch: int = 10000,
                 patience: int = None, device: torch.device = None,
                 num_cpu_workers: int = None, taskman_managed: bool = None,
                 seed: int = None, use_gpu: bool = None,
                 worker_interpolation: bool = None,
                 comet_workspace: str = None, comet_project: str = None,
                 from_checkpoint: bool = False, **_):
        """
        Parameters
        ----------
        batch_sampler_training: BatchSequencesSamplerOneInputVolume
            Instantiated class used for sampling batches of training data.
            Data in batch_sampler_training.source_data must be already loaded.
        batch_sampler_validation: BatchSequencesSamplerOneInputVolume
            Instantiated class used for sampling batches of validation
            data. Data in batch_sampler_training.source_data must be already
            loaded.
        model: torch.nn.module
            Instatiated class containing your model.
        experiment_path: str
            Path where to save this experiment's results and checkpoints.
            Will be saved in experiment_path/experiment_name.
        experiment_name: str
            Name of this experiment. This will also be the name that will
            appear online for comet.ml experiment.
        learning_rate: float
            Learning rate. Default: None (torch's default is 0.001)
        weight_decay: float
            Add a weight decay penalty on the parameters. Default: None.
            (torch's default is 0.01).
        max_epochs: int
            Maximum number of epochs. Note that we supervise this maximum to
            make sure it is not too big for general usage. Epochs lengths will
            be capped at max_batches_per_epoch.
        max_batches_per_epoch: int
            Maximum number of batches per epoch. Default: 10000.
        patience: int
            Use early stopping. Defines the number of epochs after which the
            model should stop if the loss hasn't improved.
        num_cpu_workers: int
            Number of parallel CPU workers.
        taskman_managed: bool
            If True, taskman manages the experiment. Do not output progress
            bars and instead output special messages for taskman.
        seed: int
            Random experiment seed.
        use_gpu: bool
            If true, use GPU device when possible instead of CPU.
        worker_interpolation: bool
            If true, ?
        comet_workspace: str
            Your comet workspace. If None, comet.ml will not be used. See our
            docs/Getting Started for more information on comet and its API key.
        comet_project: str
             Optional. Send your experiment to a specific comet.ml project.
             Otherwise will be sent to Uncategorized Experiments.
        """
        # To developpers: do not forget that changes here must be reflected
        # in the save_checkpoint method!

        # Experiment
        if not os.path.isdir(experiment_path):
            raise NotADirectoryError("The experiment path does not exist! "
                                     "({})".format(experiment_path))
        if from_checkpoint:
            self.experiment_dir = experiment_path
        else:
            self.experiment_dir = os.path.join(experiment_path,
                                               experiment_name)
            if not os.path.isdir(self.experiment_dir):
                logging.info('Creating directory {}'
                             .format(self.experiment_dir))
                os.mkdir(self.experiment_dir)
        self.experiment_name = experiment_name

        # Data
        # Note that the training/validation sets are contained in the
        # batch_samplers.data_source
        self.train_batch_sampler = batch_sampler_training
        self.valid_batch_sampler = batch_sampler_validation
        self.model = model

        # Training/validation epochs:
        self.max_epochs = max_epochs
        self.max_batches_per_epochs = max_batches_per_epoch
        # Real nb batches per epoch will be defined later
        self.nb_train_batches_per_epoch = None
        self.nb_valid_batches_per_epoch = None
        self.patience = patience
        if patience:
            self.early_stopping = EarlyStopping(patience=self.patience)
        else:
            self.early_stopping = None
        self.best_epoch = None
        self.current_epoch = 0

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
            'update_loss': None,
            'time': 0.
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

        # A voir....
        self.use_gpu = use_gpu
        self.worker_interpolation = worker_interpolation
        if self.use_gpu:
            torch.cuda.manual_seed(self.seed)  # Says error, but ok.

        # Setup monitors
        self.train_loss_monitor = ValueHistoryMonitor("Training loss")
        self.valid_loss_monitor = ValueHistoryMonitor("Validation loss")
        self.grad_norm_monitor = ValueHistoryMonitor("Grad Norm")

        # Prepare comet
        # Will be instantiated in train().
        self.comet_workspace = comet_workspace
        self.comet_project = comet_project
        self.comet_exp = None
        self.comet_key = None

        # Prepare optimizer
        # Send model to device
        # NOTE: This ordering is important! The optimizer needs to use the cuda
        # Tensors if using the GPU...
        self.model.to(device=self.device)

        # Build optimizer (Optimizer is built here since it needs the model
        # parameters)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay)

    @property
    def attributes(self) -> dict:
        """
        Return experiment attributes (anything that is not a hyperparameter).
        """
        attrs = {
            'experiment_dir': self.experiment_dir,
            'experiment_name': self.experiment_name,
            'dwi_ml_trainer_version': VERSION,
            'comet_key': self.comet_key,
            'train_hdf5_path': self.train_batch_sampler.data_source.hdf5_path,
            'valid_hdf5_path': self.valid_batch_sampler.data_source.hdf5_path,
            'train_sampler_type': type(self.train_batch_sampler),
            'valid_sampler_type': type(self.valid_batch_sampler),
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
        }
        attrs.update(self.hyperparameters)
        return attrs

    @property
    def hyperparameters(self) -> dict:
        """
        Return experiment hyperparameters in a dictionary

        You should add your model's hyperparameter in your child class
        """
        hyperparameters = {
            'max_epochs': self.max_epochs,
            'nb_training_batches_per_epoch': self.nb_train_batches_per_epoch,
            'nb_validation_batches_per_epoch':
                self.nb_valid_batches_per_epoch,
            'training_sampler': self.train_batch_sampler.hyperparameters,
            'validation_sampler': self.valid_batch_sampler.hyperparameters,
            'seed': self.seed,
            'patience': self.patience
        }
        return hyperparameters

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
                self.comet_exp.log_parameters(self.attributes)
                self.comet_exp.log_parameters(self.hyperparameters)
                self.comet_key = self.comet_exp.get_key()
        except ConnectionError:
            logging.warning("Could not connect to Comet.ml, metrics will not "
                            "be logged online...")
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

    def train_validate_and_save_loss(self, *args):
        """
        Train + validates the model

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
        # If data comes from checkpoint, this is already computed
        if self.nb_train_batches_per_epoch is None:
            (self.nb_train_batches_per_epoch,
             self.nb_valid_batches_per_epoch) = self.estimate_nb_batches_per_epoch()

        logging.info("Experiment attributes : \n{}".format(
            json.dumps(self.attributes, indent=4, sort_keys=True,
                       default=(lambda x: str(x)))))
        logging.info("Experiment hyperparameters: \n{}".format(
            json.dumps(self.hyperparameters, indent=4, sort_keys=True,
                       default=(lambda x: str(x)))))

        # Instantiate comet experiment
        # If self.comet_key is None: new experiment, will create a key
        # Else, resuming from checkpoint. Will continue with given key.
        self._init_comet()
        if self.comet_exp:
            train_context = self.comet_exp.train_validate_and_save_loss
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
        train_dataloader = DataLoader(
            self.train_batch_sampler.data_source,
            batch_sampler=self.train_batch_sampler,
            num_workers=self.num_cpu_workers,
            collate_fn=self.train_batch_sampler.load_batch,
            pin_memory=self.use_gpu and self.worker_interpolation)

        valid_dataloader = DataLoader(
            self.valid_batch_sampler.data_source,
            batch_sampler=self.valid_batch_sampler,
            num_workers=self.num_cpu_workers,
            collate_fn=self.valid_batch_sampler.load_batch,
            pin_memory=self.use_gpu and self.worker_interpolation)

        # Instantiating our IterTimer. DOES WHAT???
        iter_timer = IterTimer(history_len=20)

        # Start from current_spoch in case the experiment is resuming
        # Train each epoch
        for epoch in iter_timer(range(self.current_epoch, self.max_epochs)):
            self.current_epoch = epoch
            logging.info("Epoch #{}".format(epoch))

            # Training
            self.train_one_epoch(train_dataloader, train_context, epoch,
                                 *args)

            # Validation
            self.validate_one_epoch(valid_dataloader, valid_context, epoch,
                                    *args)

            # Check for early stopping
            if self.early_stopping.step(
                    self.valid_loss_monitor.epochs_means_history[-1]):
                self.save_checkpoint()
                raise EarlyStoppingError(
                    "Early stopping! Loss has not improved after {} epochs!\n"
                    "Best result: {}; At epoch #{}"
                    .format(self.patience,
                            self.early_stopping.best, self.best_epoch))

            # Check for current best
            if self.valid_loss_monitor.epochs_means_history[-1] < (
                    self.early_stopping.best + self.early_stopping.min_eps):
                logging.info("Best epoch yet! Saving model and loss history.")
                self.model.best_model_state = self.model.state_dict()

                # Save in path
                self.model.save(self.experiment_dir)

                # Save losses (i.e. mean over all batches)
                self.best_epoch = self.current_epoch
                losses = {
                    'train_loss': self.train_loss_monitor.epochs_means_history[
                        self.best_epoch],
                    'valid_loss': self.early_stopping.best}
                with open(os.path.join(self.experiment_dir, "losses.json"),
                          'w') as json_file:
                    json_file.write(
                        json.dumps(losses, indent=4, separators=(',', ': ')))

                # Save information online
                if self.comet_exp:
                    self.comet_exp.log_metric("best_validation",
                                              self.early_stopping.best)
                    self.comet_exp.log_metric("best_epoch",
                                              self.best_epoch)

            # End of epoch, save checkpoint for resuming later
            self.save_checkpoint()

            if self.taskman_managed:
                updates = {
                    'loss_train':
                        self.train_loss_monitor.epochs_means_history[-1],
                    'loss_valid':
                        self.valid_loss_monitor.epochs_means_history[-1],
                    'epoch': self.current_epoch,
                    'best_epoch': self.best_epoch,
                    'best_loss': self.early_stopping.best
                }
                self._update_taskman_report(updates)

            # (For taskman) Check if time is running out
            if self._should_quit(iter_timer):
                print('Seems like I should quit, so I quit.')
                print('Remaining: {:.0f} s'.format(
                    self.hangup_time - time.time()))
                self._update_taskman_report({'resubmit': True})
                exit(2)

        # Training is over, save checkpoint
        self.save_checkpoint()

    def train_one_epoch(self, train_dataloader, train_context, epoch, *args):
        """
        Train one epoch of the model: loop on all batches.

        All *args will be passed all to run_one_batch, which you should
        implement, in case you need some variables.
        """
        # Make sure there are no existing HDF handles if using parallel workers
        if (self.num_cpu_workers > 0 and
                self.train_batch_sampler.data_source.is_lazy):
            self.train_batch_sampler.data_source.hdf_handle = None
            self.train_batch_sampler.data_source.volume_cache_manager = None

        if self.comet_exp:
            self.comet_exp.log_metric("current_epoch", self.current_epoch)

        # Training all batches
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
                        batch_sampler=self.train_batch_sampler, *args)

                    logging.debug("Updated loss: {}".format(mean_loss))
                    self.train_loss_monitor.update(mean_loss)
                    self.grad_norm_monitor.update(grad_norm)

                    # Update taskman every 10 updates
                    if self.taskman_managed and batch_id % 10 == 0:
                        th = self.train_loss_monitor.epochs_means_history
                        vh = self.valid_loss_monitor.epochs_means_history
                        updates = {
                            'loss_train': th[-1] if len(th) > 0 else None,
                            'loss_valid': vh[-1] if len(vh) > 0 else None,
                            'epoch': self.current_epoch,
                            'best_epoch': self.best_epoch,
                            'best_loss': self.early_stopping.best,
                            'update': batch_id,
                            'update_loss': mean_loss
                        }
                        self._update_taskman_report(updates)

                    # Update Comet every 10 updates
                    if self.comet_exp and batch_id % 10 == 0:
                        self.comet_exp.log_metric(
                            "loss_step",
                            mean_loss,
                            step=batch_id)
                        self.comet_exp.log_metric(
                            "gradient_norm_step",
                            self.grad_norm_monitor.current_epoch_history[-1],
                            step=batch_id)

            # Explicitly delete iterator to kill threads and free memory before
            # running validation
            del train_iterator

        # Saving epoch's information
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

        logging.info("Mean gradient norm : {}"
                     .format(self.grad_norm_monitor.epochs_means_history[-1]))
        logging.info("Mean training loss : {}"
                     .format(self.train_loss_monitor.epochs_means_history[-1]))

    def validate_one_epoch(self, valid_dataloader, valid_context, epoch,
                           *args):
        """
        Validate one epoch of the model: loop on all batches.

        All *args will be passed all to run_one_batch, which you should
        implement, in case you need some variables.
        """
        logging.debug('Unused args in validate: {}'.format(args))

        # Make sure there are no existing HDF handles if using parallel workers
        if (self.num_cpu_workers > 0 and
                self.valid_batch_sampler.data_source.is_lazy):
            self.valid_batch_sampler.data_source.hdf_handle = None
            self.valid_batch_sampler.data_source.volume_cache_manager = None

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
                mean_loss = self.run_one_batch(
                    data, is_training=False,
                    batch_sampler=self.valid_batch_sampler, *args)
                self.valid_loss_monitor.update(mean_loss)

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
        logging.info("Validation loss : {}"
                     .format(self.valid_loss_monitor.epochs_means_history[-1]))

    def run_one_batch(self, data, is_training: bool, batch_sampler, *args):
        """
        PLEASE DEFINE IN YOUR CHILD CLASS

        Run a batch of data through the model (calling its forward method)
        and return the mean loss. If training, run the backward method too.

        In the trainer, this is called inside the loop:
        for epochs: (goes in train_one_epoch)
            for batches: (goes in train_one_batch)
                self.model.run_model_and_compute_loss

        Hint 1: With torch models, you can use self.model.train() or
        self.model.eval() to allow gradients modification or not.

        Hint 2: If your sampler was instantiated with avoid_cpu_computations,
        you need to deal with your data accordingly here!
        Use the sampler's self.load_batch_final_step method.

        Returns
        -------
        mean_loss: float
            The mean loss for this batch
        grad_norm: float
            The total grad_norm (total norm: sqrt(sum(params**2)))
            for this batch.
        """
        # logging.debug('Unused args in train: {}'.format(args))

        raise NotImplementedError

    @classmethod
    def init_from_checkpoint(cls, batch_sampler_training: BatchSamplerAbstract,
                             batch_sampler_validation: BatchSamplerAbstract,
                             model: torch.nn.module,
                             checkpoint_state: dict):
        """
        During save_checkpoint(), checkpoint_state.pkl is saved. Loading it
        back offers a dict that can be used to instantiate an experiment and
        set it at the same state as previously. (Current_epoch is updated +1).

        Hint: If you want to use this in your child class, use:
        experiment, checkpoint_state = super(cls, cls).init_from_checkpoint(...
        """
        experiment = cls(batch_sampler_training, batch_sampler_validation,
                         model, from_checkpoint=True, **checkpoint_state)

        # Set RNG states
        torch.set_rng_state(checkpoint_state['torch_rng_state'])
        experiment.rng.set_state(checkpoint_state['numpy_rng_state'])
        if (experiment.use_gpu and
                checkpoint_state['torch_cuda_state'] is not None):
            torch.cuda.set_rng_state(checkpoint_state['torch_cuda_state'])

        # Set other objects
        experiment.best_epoch = checkpoint_state['best_epoch']
        experiment.comet_key = checkpoint_state['comet_key']
        experiment.current_epoch = checkpoint_state['current_epoch'] + 1
        experiment.nb_train_batches_per_epoch = \
            checkpoint_state['nb_train_batches_per_epoch']
        experiment.nb_valid_batches_per_epoch = \
            checkpoint_state['nb_valid_batches_per_epoch']
        experiment.early_stopping.set_state(
            checkpoint_state['early_stopping_state'])
        experiment.train_loss_monitor.set_state(
            checkpoint_state['train_loss_monitor_state'])
        experiment.valid_loss_monitor.set_state(
            checkpoint_state['valid_loss_monitor_state'])
        experiment.grad_norm_monitor.set_state(
            checkpoint_state['grad_norm_monitor_state'])
        experiment.optimizer.load_state_dict(
            checkpoint_state['optimizer_state'])
        return experiment, checkpoint_state

    def save_checkpoint(self):
        """
        Save an experiment checkpoint that can be resumed from.
        """
        # Make model directory
        checkpoint_dir = os.path.join(self.experiment_dir, "checkpoint")

        # Backup old checkpoint before saving, and erase it afterwards
        to_remove = None
        if os.path.exists(checkpoint_dir):
            to_remove = os.path.join(self.experiment_dir, "checkpoint_old")
            shutil.move(checkpoint_dir, to_remove)

        os.mkdir(checkpoint_dir)

        # Save experiment
        checkpoint_state = self._prepare_checkpoint_state()
        torch.save(checkpoint_state,
                   os.path.join(checkpoint_dir, "checkpoint_state.pkl"))

        if to_remove:
            shutil.rmtree(to_remove)

    def _prepare_checkpoint_state(self) -> dict:
        # These are the parameters necessary to load data (see train_model.py)
        checkpoint_state = {
            'train_sampler_params': self.train_batch_sampler.attributes,
            'valid_sampler_params': self.valid_batch_sampler.attributes,
            'train_data_params':
                self.train_batch_sampler.data_source.attributes,
            'valid_data_params':
                self.valid_batch_sampler.data_source.attributes,
            'model_params': self.model.attributes,
        }

        # These are the parameters necessary to use _init_
        params_for_init = {
            'experiment_path': self.experiment_dir,
            'experiment_name': self.experiment_name,
            'max_epochs': self.max_epochs,
            'max_batches_per_epoch': self.max_batches_per_epochs,
            'patience': self.patience,
            'device': self.device,
            'num_cpu_workers': self.num_cpu_workers,
            'taskman_managed': self.taskman_managed,
            'seed': self.seed,
            'use_gpu': self.use_gpu,
            'worker_interpolation': self.worker_interpolation,
            'comet_workspace': self.comet_project,
            'comet_project': self.comet_project
        }

        # These are parameters that should be updated after instantiating cls.
        other_params = {
            'best_epoch': self.best_epoch,
            'comet_key': self.comet_key,
            'current_epoch': self.current_epoch,
            'nb_train_batches_per_epoch': self.nb_train_batches_per_epoch,
            'nb_valid_batches_per_epoch': self.nb_valid_batches_per_epoch,
            'torch_rng_state': torch.random.get_rng_state(),
            'torch_cuda_state':
                torch.cuda.get_rng_state() if self.use_gpu else
                None,
            'numpy_rng_state': self.rng.get_state(),
            'early_stopping_state':
                self.early_stopping.get_state() if self.early_stopping else
                None,
            'train_loss_monitor_state': self.train_loss_monitor.get_state(),
            'valid_loss_monitor_state': self.valid_loss_monitor.get_state()
        }

        checkpoint_state.update(params_for_init)
        checkpoint_state.update(other_params)
        return checkpoint_state

    def _should_quit(self, iter_timer):
        # If:
        #   - hang up signal received
        #   - time remaining is less than one epoch + 30 seconds
        # exit training.
        return (self.hangup_time is not None and
                time.time() + iter_timer.mean * 2.0 + 30 > self.hangup_time)

    def _update_taskman_report(self, updates):
        self.taskman_report.update(updates)
        self.taskman_report['time'] = time.time()
        print('!taskman' + json.dumps(self.taskman_report), flush=True)

    def _save_log_from_array(self, array: np.ndarray, fname: str):
        log_dir = os.path.join(self.experiment_dir, "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        fpath = os.path.join(log_dir, fname)
        np.save(fpath, array)

    @staticmethod
    def load_params_from_checkpoint(experiment_path: str):
        total_path = os.path.join(
            experiment_path, "checkpoint", "checkpoint_state.pkl")
        if not os.path.isfile(total_path):
            raise FileNotFoundError('Checkpoint was not found! ({})'
                                    .format(total_path))
        checkpoint_state = torch.load(total_path)

        return checkpoint_state

    @staticmethod
    def check_early_stopping(checkpoint_state, new_patience=None):
        msg = "Resumed experiment was stopped because of early stopping, " \
              "increase patience in order to resume training!"

        if new_patience is None:
            if (checkpoint_state['early_stopping_state']['n_bad_epochs'] >=
                    checkpoint_state['early_stopping_state']['patience']):
                raise EarlyStoppingError(msg)
        else:
            if (checkpoint_state['early_stopping_state']['n_bad_epochs'] >=
                    new_patience):
                raise EarlyStoppingError(msg)
