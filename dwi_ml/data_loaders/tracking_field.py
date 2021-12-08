# -*- coding: utf-8 -*-
import numpy as np
from dwi_ml.data.processing.space.neighborhood import \
    prepare_neighborhood_information
from torch.nn.utils.rnn import pack_sequence

from dwi_ml.data.dataset.single_subject_containers import SubjectDataAbstract
from dwi_ml.models.main_models import MainModelAbstract
from dwi_ml.data.processing.streamlines.post_processing import \
    compute_n_previous_dirs


class DWIMLAbstractTrackingField:
    """
    Abstract tracking field. The class is used to get outputs from the model.

    Similar to scilpy.tracking.tracking_field. However, due to major
    differences, we are not using their class as super. Created to be used
    with our DWIMLPropagator. Differences with traditional tractography:
        - current data values (ex, fODF, SH) are not known. The method needs to
        get the current data from the hdf5 and prepare it as an input with
        format expected by the model.
        - the output of the model is not necessarily a direction. It could be,
        for instance, the logits associated with the classes in a
        classification model. Or it could be the means and sigmas representing
        the learned gaussian. Each model should contain methods to sample a
        direction based on these outputs. (sampling is done in the propagator)
        - theta would be very complex to include here (to restrain the list of
        possible directions, as they depend on the model) and should rather be
        used as stopping criteria, later.

    Acts similarly to the batch sampler during training.
    """
    def __init__(self, model: MainModelAbstract,
                 subj_data: SubjectDataAbstract, device=None):
        """
        Parameters
        ----------
        model: MainModelAbstract
             A learned model.
        subj_data: SubjectDataAbstract
            Either LazySubjectData or SubjectData. An instance of the data for
            a subject.
        """
        self.model = model
        self.subj_data = subj_data

        # Everything tracking and in scilpy.tracking is in 'corner', 'voxmm'
        self.origin = 'corner'
        self.space = 'voxmm'

        self.device = device
        if device is not None:
            self.move_to(device)

    def move_to(self, device):
        self.model.to(device=device)
        self.device = device

    def get_model_outputs_at_pos(self, pos, *args):
        """
        Runs the model (calls the forward method) to get the outputs at
        current position.

        Parameters
        ----------
        pos: ndarray (3,)
            Position in trackable dataset, expressed in mm.
        """
        # Something like
        # inputs = self._get_inputs_at_pos(pos)
        # model_outputs = self.model.forward(inputs)
        raise NotImplementedError


class DWIMLTrackingFieldOneInputAndPD(DWIMLAbstractTrackingField):
    """
    This version of the tracking field is for when data is represented as
    x: one volume input (formatted through the model, possibly with
       neighborhood).
    p: the n previous directions of the streamline.

    The class is used to get outputs from the model.

    This is in fact similar to our batch sampler (with inputs): it needs to get
    the data points from the volume (+possibly add a neighborhood) and
    interpolate the data.
    """
    def __init__(self, model: MainModelAbstract,
                 subj_data: SubjectDataAbstract, input_volume_group: str,
                 neighborhood_type, neighborhood_radius):
        """
        Parameters
        ----------
        model: MainModelAbstract
             A learned model.
        subj_data: SubjectDataAbstract
            Either LazySubjectData or SubjectData. An instance of the data for
            a subject.
        input_volume_group: str
            The volume group to use as input in the model.
        """
        super().__init__(model, subj_data)
        self.volume_group_str = input_volume_group
        self.neighborhood_type = neighborhood_type
        self.neighborhood_radius = neighborhood_radius

        # Preparing the neighborhood
        self.neighborhood_points = prepare_neighborhood_information(
            neighborhood_type, neighborhood_radius)

        # Find group index in the data
        self.volume_group = subj_data.volume_groups.index(input_volume_group)

    def get_model_outputs_at_pos(self, pos, previous_dirs=None):
        """
        Parameters
        ----------
        pos: ndarray (3,)
            Current position coordinates.
        previous_dirs: 3D coordinates
            List of the previous directions
            Size: (length of the streamline - 1) x 3. If this is the first step
            of the streamline, use None.
        """
        inputs = self.subj_data.mri_data_list[self.volume_group]

        # Get pos in voxel world
        pos_vox = inputs.as_data_volume.voxmm_to_vox(*pos, self.origin)

        # torch trilinear interpolation uses origin='corner'
        if self.origin == 'center':
            pos_vox += 0.5

        inputs_arranged, _, = self.model.prepare_inputs(
            inputs.as_tensor, np.asarray([pos_vox]), self.device)
        n_previous_dirs = self.model.prepare_previous_dirs(previous_dirs,
                                                           self.device)

        # Packing data
        inputs_packed = pack_sequence([inputs_arranged], enforce_sorted=False)
        if len(n_previous_dirs) > 0:
            n_previous_dirs = pack_sequence(n_previous_dirs,
                                            enforce_sorted=False)
        else:
            n_previous_dirs = None

        model_outputs = self.model.forward(inputs_packed, n_previous_dirs)

        return model_outputs
