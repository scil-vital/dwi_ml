import logging

from dwi_ml.models.main_models import MainModelAbstract


class Dir2Vec(MainModelAbstract):
    def __init__(self, experiment_name: str,
                 log_level=logging.root.level):
        super().__init__(experiment_name, log_level)

    def compute_loss(self, *model_outputs, **kw):
        raise NotImplementedError

    def forward(self, inputs):
        """
        Args
        ----
        inputs: Tensor
            List of streamlines.
        """
        raise NotImplementedError
