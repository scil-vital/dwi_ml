# -*- coding: utf-8 -*-
import os

from dwi_ml.models.projects.projects_list import model_classes


def verify_checkpoint_exists(experiment_path):
    checkpoint_path = os.path.join(experiment_path, "checkpoint")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Experiment's checkpoint not found ({})."
                                .format(checkpoint_path))
    return checkpoint_path


def verify_which_model_in_path(model_dir):

    model_type_txt = os.path.join(model_dir, 'model_type.txt')

    with open(model_type_txt, 'r') as txt_file:
        model_type = txt_file.readlines()

    model_type = model_type[0].replace('\n', '')
    if model_type not in model_classes:
        raise ValueError("Model type saved in {} does not correspond to any "
                         "current model among {}."
                         .format(model_type_txt, model_classes))

    model_class = model_classes[model_type]

    return model_type, model_class
