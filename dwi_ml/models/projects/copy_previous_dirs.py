# -*- coding: utf-8 -*-
from typing import List

import torch
from torch.distributions import Categorical

from dwi_ml.data.processing.streamlines.post_processing import \
    compute_directions
from dwi_ml.data.processing.streamlines.sos_eos_management import \
    convert_dirs_to_class
from dwi_ml.models.main_models import ModelWithDirectionGetter


class CopyPrevDirModel(ModelWithDirectionGetter):
    def __init__(self, dg_key: str = 'cosine-regression',
                 dg_args: dict = None, skip_first_point=False,
                 step_size=None, compress=None):
        super().__init__(dg_key=dg_key, dg_args=dg_args,
                         experiment_name='TEST', step_size=step_size,
                         compress=compress)

        # Fake input size, we won't use the forward method.
        self.instantiate_direction_getter(dg_input_size=1)
        self.skip_first_point = skip_first_point

    def forward(self, inputs, streamlines, **kw):
        # Prepare targets and outputs: both out of directions.
        # Similar to direction_getter.prepare_targets:
        # A) Compute directions. Shift + add a fake first direction.

        # Ignoring inputs.
        targets = compute_directions(streamlines)
        if self.skip_first_point:
            outputs = targets
        else:
            # Add fake first point.
            zeros = torch.as_tensor([0., 0., 0.], dtype=torch.float32,
                                    device=self.device)
            outputs = [torch.vstack((zeros, t)) for t in targets]

        # Faking outputs based on direction getter format
        if 'classification' in self.dg_key:
            outputs = convert_dirs_to_class(outputs,
                                            self.direction_getter.torch_sphere)
            outputs = torch.hstack(outputs)

            # Prepare logits_per_class
            outputs = torch.nn.functional.one_hot(outputs.to(dtype=torch.long))

            distribution = Categorical(probs=outputs)
            outputs = distribution.logits
        elif 'regression' in self.dg_key:
            outputs = torch.vstack(outputs)
        else:
            raise NotImplementedError

        return outputs

    def compute_loss(self, model_outputs: torch.Tensor,
                     target_streamlines: List[torch.Tensor],
                     average_results=True, **kw):
        targets = self.direction_getter.prepare_targets_for_loss(
            target_streamlines)

        if self.skip_first_point:
            targets = [t[1:] for t in targets]

        if self.dg_key == 'sphere-classification':
            targets = torch.hstack(targets)
        else:
            targets = torch.vstack(targets)

        if self._context == 'visu':
            return self.direction_getter.compute_loss(
                model_outputs, targets, average_results)
        else:
            raise NotImplementedError
