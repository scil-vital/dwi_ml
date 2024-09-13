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
    def __init__(self, dg_key: str = 'cosine-regression', dg_args: dict = None,
                 skip_first_point=False, step_size=None, compress_lines=None):
        """
        Fake model, not very useful. Used to test value of a loss when copying
        the previous direction.
        """
        super().__init__(dg_key=dg_key, dg_args=dg_args,
                         experiment_name='TEST',
                         step_size=step_size, compress_lines=compress_lines)

        # Fake input size, we won't use the forward method.
        self.instantiate_direction_getter(dg_input_size=1)
        self.skip_first_point = skip_first_point

    def forward(self, inputs, streamlines, **kw):
        # Prepare targets and outputs: both out of directions.
        # Similar to direction_getter.prepare_targets:
        # A) Compute directions. Shift + add a fake first direction.

        # Ignoring inputs.
        outputs = compute_directions(streamlines)

        # Output at first position will be 0. We will add it later.
        # Faking outputs based on direction getter format
        if 'classification' in self.dg_key:
            outputs = convert_dirs_to_class(
                outputs, self.direction_getter.torch_sphere, add_sos=False,
                add_eos=self.direction_getter.add_eos, to_one_hot=True)
            outputs = [Categorical(probs=out).logits for out in outputs]
        elif not ('regression' in self.dg_key):
            raise NotImplementedError

        if not self.skip_first_point:
            # Add fake first point. We don't know what output to give. Just
            # using zeros everywhere.
            outputs = [torch.nn.functional.pad(out, [0, 0, 1, 0])
                       for out in outputs]

        return outputs

    def compute_loss(self, model_outputs: List[torch.Tensor],
                     target_streamlines: List[torch.Tensor],
                     average_results=True, return_eos_probs=False):
        if self.skip_first_point:
            target_streamlines = [t[1:] for t in target_streamlines]

        return self.direction_getter.compute_loss(
            model_outputs, target_streamlines, average_results,
            return_eos_probs)
