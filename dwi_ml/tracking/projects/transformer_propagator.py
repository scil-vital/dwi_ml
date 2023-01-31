# -*- coding: utf-8 -*-
from dwi_ml.data.dataset.multi_subject_containers import MultisubjectSubset
from dwi_ml.models.projects.transforming_tractography import AbstractTransformerModel
from dwi_ml.tracking.propagator import (DWIMLPropagatorOneInput,
                                        DWIMLPropagatorwithStreamlineMemory)


class TransformerPropagator(DWIMLPropagatorOneInput,
                            DWIMLPropagatorwithStreamlineMemory):
    """
    For the Transformer, we simply need OneInput + StreamlineMemory.
    """
    model: AbstractTransformerModel

    def __init__(self, dataset: MultisubjectSubset, subj_idx: int,
                 model: AbstractTransformerModel, input_volume_group: str,
                 step_size: float, algo: str, theta: float,
                 device=None):
        super().__init__(input_volume_group=input_volume_group,
                         dataset=dataset, subj_idx=subj_idx, model=model,
                         step_size=step_size, algo=algo, theta=theta,
                         device=device,
                         # Always fixed for Transformers:
                         verify_opposite_direction=False,
                         input_memory=True)

    def _call_model_forward(self, inputs, lines):
        import logging
        logging.warning("Inputs: {}".format((len(inputs), inputs[0].shape)))
        logging.warning("Lines: {}".format((len(lines), lines[0].shape)))
        return self.model(inputs, lines, is_tracking=True)
