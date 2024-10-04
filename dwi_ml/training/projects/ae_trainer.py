# -*- coding: utf-8 -*-
import logging
import os
from typing import Union, List

from dwi_ml.models.main_models import MainModelAbstract
from dwi_ml.training.batch_loaders import DWIMLStreamlinesBatchLoader
from dwi_ml.training.batch_samplers import DWIMLBatchIDSampler
from dwi_ml.training.trainers import DWIMLAbstractTrainer
from dwi_ml.viz.latent_streamlines import BundlesLatentSpaceVisualizer


def parse_bundle_mapping(bundles_mapping_file: str = None):
    if bundles_mapping_file is None:
        return None

    with open(bundles_mapping_file, 'r') as f:
        bundle_mapping = {}
        for line in f:
            bundle_name, bundle_number = line.strip().split()
            bundle_mapping[int(bundle_number)] = bundle_name
    return bundle_mapping


class TrainerWithBundleDPS(DWIMLAbstractTrainer):

    def __init__(self,
                 model: MainModelAbstract, experiments_path: str,
                 experiment_name: str, batch_sampler: DWIMLBatchIDSampler,
                 batch_loader: DWIMLStreamlinesBatchLoader,
                 learning_rates: Union[List, float] = None,
                 weight_decay: float = 0.01,
                 optimizer: str = 'Adam', max_epochs: int = 10,
                 max_batches_per_epoch_training: int = 1000,
                 max_batches_per_epoch_validation: Union[int, None] = 1000,
                 patience: int = None, patience_delta: float = 1e-6,
                 nb_cpu_processes: int = 0, use_gpu: bool = False,
                 clip_grad: float = None,
                 comet_workspace: str = None, comet_project: str = None,
                 from_checkpoint: bool = False, log_level=logging.root.level,
                 ls_viz_freq: int = None, color_by: str = None,
                 bundles_mapping_file: str = None, max_viz_subset_size: int = 1000):

        super().__init__(model, experiments_path, experiment_name, batch_sampler, batch_loader,
                         learning_rates, weight_decay, optimizer, max_epochs, max_batches_per_epoch_training,
                         max_batches_per_epoch_validation, patience, patience_delta, nb_cpu_processes, use_gpu,
                         clip_grad, comet_workspace, comet_project, from_checkpoint, log_level)

        self.ls_viz_freq = ls_viz_freq
        self.color_by = color_by
        self.visualize = ls_viz_freq is not None
        if self.visualize:
            # Setup to visualize latent space
            save_dir = os.path.join(
                experiments_path, experiment_name, 'latent_space_plots')
            os.makedirs(save_dir, exist_ok=True)

            bundle_mapping = parse_bundle_mapping(bundles_mapping_file)
            self.ls_viz = BundlesLatentSpaceVisualizer(save_dir,
                                                       prefix_numbering=True,
                                                       max_subset_size=max_viz_subset_size,
                                                       bundle_mapping=bundle_mapping)

            # Register what to do post encoding.
            def visualize_latent_space(encoding, data_per_streamline):
                # Only execute the following if we are in training
                if not self.model.context == 'training':
                    return

                if self.color_by is None:
                    bundle_index = None
                else:
                    bundle_index = \
                        data_per_streamline[self.color_by].squeeze(1)

                self.ls_viz.add_data_to_plot(encoding, labels=bundle_index)
            model.register_hook_post_encoding(visualize_latent_space)

    def train_one_epoch(self, epoch):
        super().train_one_epoch(epoch)

        # Do things post epoch
        if self.visualize:
            current_epoch = self.current_epoch
            best_epoch = self.best_epoch_monitor.best_epoch \
                if self.best_epoch_monitor.best_epoch is not None else current_epoch
            if current_epoch % self.ls_viz_freq == 0:
                self.ls_viz.plot(current_epoch, best_epoch=best_epoch)
                self.ls_viz.reset_data()
            else:

                # TODO: This is problematic, the best epoch is updated actually after
                # validation (which is after doing the training epoch)

                self.ls_viz.check_and_register_best_epoch(
                    current_epoch, best_epoch=best_epoch)
