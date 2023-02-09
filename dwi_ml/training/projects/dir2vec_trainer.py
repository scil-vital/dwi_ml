# -*- coding: utf-8 -*-
import logging

from dwi_ml.training.trainers import DWIMLAbstractTrainer

logger = logging.getLogger('trainer_logger')


class Dir2VectTrainer(DWIMLAbstractTrainer):
    """
    Trainer for Learn2Track. Nearly the same as in dwi_ml, but we add the
    clip_grad parameter to avoid exploding gradients, typical in RNN.
    """
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

