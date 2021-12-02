# -*- coding: utf-8 -*-
import json
import logging
import os
import shutil

import torch

from dwi_ml.experiment_utils.prints import TqdmLoggingHandler


class MainModelAbstract(torch.nn.Module):
    """
    To be used for all models that will be trained. Defines the way to save
    the model.

    It should also define a forward() method.
    """
    def __init__(self, experiment_name='my_model'):
        """
        Params
        ------
        experiment_name: str
            Name of the experiment
        """
        super().__init__()

        self.experiment_name = experiment_name
        self.best_model_state = None

        # Model's logging level can be changed separately from main scripts.
        self.logger = logging.getLogger('model_logger')
        self.logger.propagate = False
        self.logger.setLevel(logging.root.level)

    def set_logger_level(self, level):
        self.logger.setLevel(level)

    def make_logger_tqdm_fitted(self):
        """Possibility to use a tqdm-compatible logger in case the model
        is used through a tqdm progress bar."""
        self.logger.addHandler(TqdmLoggingHandler())

    @property
    def params(self):
        """All parameters necessary to create again the same model. Will be
        used in the trainer, when saving the checkpoint state. Params here
        will be used to re-create the model when starting an experiment from
        checkpoint. You should be able to re-create an instance of your
        model with those params."""
        return {
            'experiment_name': self.experiment_name
        }

    @classmethod
    def init_from_checkpoint(cls, **params):
        model = cls(**params)
        return model

    def update_best_model(self):
        # Initialize best model
        # Uses torch's module state_dict.
        self.best_model_state = self.state_dict()

    def save(self, saving_dir):
        # Make model directory
        model_dir = os.path.join(saving_dir, "model")

        # If a model was already saved, back it up and erase it after saving
        # the new.
        to_remove = None
        if os.path.exists(model_dir):
            to_remove = os.path.join(saving_dir, "model_old")
            shutil.move(model_dir, to_remove)
        os.makedirs(model_dir)

        # Save attributes
        name = os.path.join(model_dir, "parameters.json")
        with open(name, 'w') as json_file:
            json_file.write(json.dumps(self.params, indent=4,
                                       separators=(',', ': ')))

        # Save model
        torch.save(self.best_model_state,
                   os.path.join(model_dir, "best_model_state.pkl"))

        if to_remove:
            shutil.rmtree(to_remove)

    def compute_loss(self, outputs, targets):
        raise NotImplementedError
