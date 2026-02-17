# -*- coding: utf-8 -*-
import json
import logging
import os
import shutil

import torch

from dwi_ml.experiment_utils.prints import format_dict_to_str
from dwi_ml.io_utils import add_resample_or_compress_arg

logger = logging.getLogger('model_logger')


class MainModelAbstract(torch.nn.Module):
    """
    To be used for all models that will be trained. Defines the way to save
    the model.

    It should also define a forward() method.
    """

    def __init__(self, experiment_name: str,
                 # Target preprocessing params for the batch loader + tracker
                 step_size: float = None,
                 nb_points: int = None,
                 compress_lines: float = False,
                 # Other
                 log_level=logging.root.level):
        """
        Params
        ------
        experiment_name: str
            Name of the experiment
        step_size : float
            Constant step size that every streamline should have between points
            (in mm). Default: None.
            The preprocessing steps are performed by the batch loader or by
            the tracker, but it probably influences strongly how the model
            performs, particularly in sequence-based models, as it changes the
            length of streamlines.
            When using an existing model in various scripts, you will often
            have the option to modify this value, but it is probably not
            recommanded.
        compress_streamlines: float
            If set, compress streamlines to that tolerance error. Cannot be
            used together with step_size. This model cannot be used for
            tracking.
        log_level: str
            Level of the model logger. Default: root's level.
        """
        super().__init__()

        self.experiment_name = experiment_name

        # Trainer's logging level can be changed separately from main
        # scripts.
        logger.setLevel(log_level)

        self.device = None

        # To tell our batch loader how to resample streamlines during training
        # (should also be the step size during tractography).
        if (step_size and compress_lines) or (step_size and nb_points) or (nb_points and compress_lines):
            raise ValueError("You may choose either resampling (step_size or nb_points)"
                             " or compressing, but not two of them or more.")
        elif step_size and step_size <= 0:
            raise ValueError("Step size can't be 0 or less!")
        elif nb_points and nb_points <= 0:
            raise ValueError("Number of points can't be 0 or less!")
            # Note. When using
            # scilpy.tracking.tools.resample_streamlines_step_size, a warning
            # is shown if step_size < 0.1 or > np.max(sft.voxel_sizes), saying
            # that the value is suspicious. Not raising the same warnings here
            # as you may be wanting to test weird things to understand better
            # your model.
        self.nb_points = nb_points
        self.step_size = step_size
        self.compress_lines = compress_lines

        # Adding a context. Most models in here act differently
        # during training (ex: no loss at the last coordinate = we skip it)
        # vs during tracking (only the last coordinate is important) vs during
        # visualisation (the whole streamline is important).
        self._context = None

    @staticmethod
    def add_args_main_model(p):
        add_resample_or_compress_arg(p)

    def set_context(self, context):
        assert context in ['training',  'validation']
        self._context = context

    @property
    def context(self):
        return self._context

    def move_to(self, device):
        """
        Careful. Calling model.to(a_device) does not influence the self.device.
        Prefer this method for easier management.
        """
        self.to(device, non_blocking=True)
        self.device = device

    @property
    def params_for_checkpoint(self):
        """All parameters necessary to create again the same model. Will be
        used in the trainer, when saving the checkpoint state. Params here
        will be used to re-create the model when starting an experiment from
        checkpoint. You should be able to re-create an instance of your
        model with those params."""
        return {
            'experiment_name': self.experiment_name,
            'step_size': self.step_size,
            'compress_lines': self.compress_lines,
            'nb_points': self.nb_points,
        }

    @property
    def computed_params_for_display(self):
        p = {}
        return p

    def save_params_and_state(self, model_dir):
        model_state = self.state_dict()

        # If a model was already saved, back it up and erase it after saving
        # the new.
        to_remove = None
        if os.path.exists(model_dir):
            to_remove = os.path.join(model_dir, "..", "model_old")
            shutil.move(model_dir, to_remove)
        os.makedirs(model_dir)

        # Save attributes
        name = os.path.join(model_dir, "parameters.json")
        with open(name, 'w') as json_file:
            json_file.write(json.dumps(self.params_for_checkpoint, indent=4,
                                       separators=(',', ': ')))

        name = os.path.join(model_dir, "model_type.txt")
        with open(name, 'w') as txt_file:
            txt_file.write(str(self.__class__.__name__))

        # Save model
        torch.save(model_state, os.path.join(model_dir, "model_state.pkl"))

        if to_remove:
            shutil.rmtree(to_remove)

    @classmethod
    def load_model_from_params_and_state(cls, model_dir,
                                         log_level=logging.WARNING):
        """
        Params
        -----
        loading_dir: path
            Path to the trained parameters, either from the latest checkpoint
            or from the best model folder. Must contain files
            - parameters.json
            - model_state.pkl
        """
        params = cls._load_params(model_dir)

        logger.setLevel(log_level)
        logger.debug("Loading model from saved parameters:" +
                     format_dict_to_str(params))
        params.update(log_level=log_level)
        model = cls(**params)

        model_state = cls._load_state(model_dir)
        model.load_state_dict(model_state)  # using torch's method

        # By default, setting to eval state. If this will be used by the
        # trainer, it will call model.train().
        model.eval()

        return model

    @classmethod
    def _load_params(cls, model_dir):
        # Load attributes and hyperparameters from json file
        params_filename = os.path.join(model_dir, "parameters.json")
        with open(params_filename, 'r') as json_file:
            params = json.load(json_file)

        return params

    @classmethod
    def _load_state(cls, model_dir):
        model_state_file = os.path.join(model_dir, "model_state.pkl")
        model_state = torch.load(model_state_file)

        return model_state

    def forward(self, inputs, streamlines):
        raise NotImplementedError

    def compute_loss(self, model_outputs, target_streamlines):
        raise NotImplementedError

    def merge_batches_outputs(self, all_outputs, new_batch):
        """
        To be used at testing time. At training or validation time, outputs are
        discarded after each batch; only the loss is measured. At testing time,
        it may be necessary to merge batches. The way to do it will depend on
        your model's format.

        Parameters
        ----------
        all_outputs: Any or None
            All previous outputs from previous batches, already combined.
        new_batch:
            The batch to merge
        """
        raise NotImplementedError

