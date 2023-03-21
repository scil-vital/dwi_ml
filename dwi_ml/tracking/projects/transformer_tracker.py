# -*- coding: utf-8 -*-
import torch

from dwi_ml.models.projects.transforming_tractography import AbstractTransformerModel
from dwi_ml.tracking.tracker import (
    DWIMLTrackerOneInput, DWIMLTrackerFromWholeStreamline)


# Just combining the two Abstract Classes of interest.
class TransformerTracker(DWIMLTrackerOneInput, DWIMLTrackerFromWholeStreamline):
    """
    For the Transformer, we simply need OneInput + StreamlineMemory.
    """
    model: AbstractTransformerModel

    def __init__(self, **kw):
        super().__init__(verify_opposite_direction=False, **kw)

    def _call_model_forward(self, inputs, lines):
        """Copying super's method but adding is_tracking=True"""

        # Adding the current input to the input memory
        if len(self.input_memory) == 0:
            self.input_memory = inputs
        else:
            # If they all had the same lengths we could concatenate
            # everything. But during backward, they don't.
            self.input_memory = \
                [torch.cat((self.input_memory[i], inputs[i]), dim=0)
                 for i in range(len(self.input_memory))]

        import logging
        logging.warning("INPUT: {}x{}".format(len(self.input_memory), self.input_memory[0].shape))
        logging.warning("Lines:{}x{}".format(len(lines), lines[0].shape))
        with self.grad_context:
            if self.model.forward_uses_streamlines:
                model_outputs = self.model(self.input_memory, lines,
                                           is_tracking=True)
            else:
                model_outputs = self.model(self.input_memory, is_tracking=True)
        return model_outputs
