# -*- coding: utf-8 -*-
import json
import os
import shutil

import torch


class ModelAbstract(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize best model
        # Uses torch's module state_dict.
        self.best_model_state = self.state_dict()

    @property
    def hyperparameters(self):
        return {}

    @property
    def attributes(self):
        """All parameters necessary to create again the same model"""
        return self.hyperparameters

    def save(self, saving_dir):
        # Make model directory
        model_dir = os.path.join(saving_dir, "model")

        # If a model was already saved, back it up and erase it after saving
        # the new.
        to_remove = None
        if os.path.exists(model_dir):
            to_remove = os.path.join(self.saving_dir, "model_old")
            shutil.move(model_dir, to_remove)
        os.makedirs(model_dir)

        # Save attributes
        with open(os.path.join(model_dir, "attributes.json"), 'w') as json_file:
            json_file.write(
                json.dumps(self.attributes, indent=4,
                           separators=(',', ': ')))

        # Save hyperparams
        with open(os.path.join(model_dir, "hyperparameters.json"),
                  'w') as json_file:
            json_file.write(json.dumps(self.hyperparameters, indent=4,
                                       separators=(',', ': ')))

        # Save model
        torch.save(self.best_model_state,
                   os.path.join(model_dir, "best_model_state.pkl"))

        if to_remove:
            shutil.rmtree(to_remove)

    def forward(self, *args):
        pass

    def compute_loss(self, outputs, targets):
        raise NotImplementedError
