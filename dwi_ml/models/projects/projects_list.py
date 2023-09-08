# -*- coding: utf-8 -*-
from dwi_ml.models.projects.learn2track_model import Learn2TrackModel
from dwi_ml.models.projects.transformer_models import \
    OriginalTransformerModel, TransformerSrcAndTgtModel

model_classes = {
    'Learn2TrackModel': Learn2TrackModel,
    'OriginalTransformerModel': OriginalTransformerModel,
    'TransformerSrcAndTgtModel': TransformerSrcAndTgtModel
}