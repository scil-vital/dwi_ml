import os
import logging
from typing import Union, List, Tuple
from sklearn.manifold import TSNE
import numpy as np
import torch
import matplotlib.pyplot as plt
from dwi_ml.viz.utils import generate_dissimilar_color_map

LOGGER = logging.getLogger(__name__)

DEFAULT_BUNDLE_NAME = 'UNK'


class ColorManager(object):
    """
    Utility class to manage the color of the bundles in the latent space.
    This way, we can have a consistent color for each bundle across
    different plots.
    """

    def __init__(self, max_num_bundles: int = 40):
        self.bundle_color_map = {}
        self.color_map = generate_dissimilar_color_map(max_num_bundles)

    def get_color(self, label: str):
        if label not in self.bundle_color_map:
            self.bundle_color_map[label] = \
                self.color_map(
                len(self.bundle_color_map))

        return self.bundle_color_map[label]


class BundlesLatentSpaceVisualizer(object):
    """
    Utility class that wraps a t-SNE projection of the latent
    space for multiple bundles. The usage of this class is
    intented as follows:
        1. Create an instance of this class,
        2. Add the latent space streamlines for each bundle
            using "add_data_to_plot" with its corresponding label.
        3. Fit and plot the t-SNE projection using the "plot" method.

    t-SNE projection can only leverage the fit_transform() with all
    the data that needs to be projected at the same time since it aims
    to preserve the local structure of the data.
    """

    def __init__(self,
                 save_dir: str,
                 fig_size: Union[List, Tuple] = (10, 8),
                 random_state: int = 42,
                 max_subset_size: int = None,
                 prefix_numbering: bool = False,
                 reset_warning: bool = True,
                 bundle_mapping: dict = None
                 ):
        """
        Parameters
        ----------
        save_path: str
            Path to save the figure. If not specified, the figure will show.
        fig_size: List[int] or Tuple[int]
            2-valued figure size (x, y)
        random_state: List
            Random state for t-SNE.
        max_subset_size: int
            In case of performance issues, you can limit the number of
            streamlines to plot for each bundle.
        prefix_numbering: bool
            If True, the saved figures will be numbered with the current
            plot number. The plot number is incremented after each call
            to the "plot" method.
        reset_warning: bool
            If True, a warning will be displayed when calling "plot"several
            times without calling "reset_data" in between to clear the data.
        bundle_mapping: dict
            Mapping of the bundle names to the labels to display on the plot.
            (e.g. key: bundle index, value: bundle name)
        """
        self.save_dir = save_dir

        # Make sure that self.save_dir is a directory and exists.
        if not os.path.isdir(self.save_dir):
            raise ValueError("The save_dir should be a directory.")

        self.fig_size = fig_size
        self.random_state = random_state
        self.max_subset_size = max_subset_size
        if not (self.max_subset_size > 0):
            raise ValueError(
                "A max_subset_size of an integer value greater"
                "than 0 is required.")

        self.prefix_numbering = prefix_numbering
        self.reset_warning = reset_warning
        self.bundle_mapping = bundle_mapping

        self.current_plot_number = 0
        self.should_call_reset_before_plot = False

        self.tsne = TSNE(n_components=2, random_state=self.random_state)
        self.bundles = {}
        self.bundle_color_manager = ColorManager()

        self.fig, self.ax = None, None

    def reset_data(self):
        """
        Reset the data to plot. If you call plot several times without
        calling this method, the data will be accumulated.
        """
        # Not sure if resetting the TSNE object is necessary.
        self.tsne = TSNE(n_components=2, random_state=self.random_state)
        self.bundles = {}
        self.should_call_reset_before_plot = False

    def add_data_to_plot(self, data: np.ndarray, labels: List[str]):
        """
        Add unprojected data (no t-SNE, no PCA, etc.).
        This should be directly the output of the model as a numpy array.

        Parameters
        ----------
        data: str
            Unprojected latent space streamlines (N, latent_space_dim).
        labels: np.ndarray
            Labels for each streamline.
        """
        latent_space_streamlines = self._to_numpy(data)
        if labels is None:
            self.bundles[DEFAULT_BUNDLE_NAME] = latent_space_streamlines
        else:
            all_labels = np.unique(labels)
            remaining_indices = np.arange(len(labels))
            for label in all_labels:
                label_indices = labels[remaining_indices] == label
                label_data = \
                    latent_space_streamlines[remaining_indices][label_indices]
                label_data = self._resample_max_subset_size(label_data)
                self.bundles[label] = label_data

                remaining_indices = remaining_indices[~label_indices]

            if len(remaining_indices) > 0:
                LOGGER.warning(
                    "Some streamlines were not considered in the bundles,"
                    "some labels are missing.\n"
                    "Added them to the {} bundle."
                    .format(DEFAULT_BUNDLE_NAME))
                self.bundles[DEFAULT_BUNDLE_NAME] = \
                    latent_space_streamlines[remaining_indices]

    def add_bundle_to_plot(self, data: np.ndarray, label: str = '_'):
        """
        Add unprojected data (no t-SNE, no PCA, etc.).
        This should be directly the output of the model as a numpy array.

        Parameters
        ----------
        data: str
            Unprojected latent space streamlines (N, latent_space_dim).
        label: str
            Name of the bundle. Used for the legend.
        """
        latent_space_streamlines = self._to_numpy(data)
        latent_space_streamlines = self._resample_max_subset_size(
            latent_space_streamlines)

        self.bundles[label] = latent_space_streamlines

    def plot(self, epoch: int, figure_name_prefix: str = 'lt_space'):
        """
        Fit and plot the t-SNE projection of the latent space streamlines.
        This should be called once after adding all the data to plot using
        "add_data_to_plot".

        Parameters
        ----------
        figure_name_prefix: str
            Name of the figure to be saved. This is just the prefix of the
            full file name as it will be suffixed with the current plot
            number if enabled.
        """
        if self.should_call_reset_before_plot and self.reset_warning:
            LOGGER.warning(
                "You plotted another time without resetting the data. "
                "The data will be accumulated, which might lead to "
                "unexpected results.")
            self.should_call_reset_before_plot = False
        elif not self.current_plot_number > 0:
            # Only enable the flag for the first plot.
            # So that the warning above is only displayed once.
            self.should_call_reset_before_plot = True

        # Start by making sure the number of streamlines doesn't
        # exceed the threshold.
        for (bname, bdata) in self.bundles.items():
            if bdata.shape[0] > self.max_subset_size:
                self.bundles[bname] = self._resample_max_subset_size(bdata)

        nb_streamlines = sum(b.shape[0] for b in self.bundles.values())
        LOGGER.info(
            "Plotting a total of {} streamlines".format(nb_streamlines))

        # Build the indices for each bundle to recover the streamlines after
        # the t-SNE projection.
        bundles_indices = {}
        current_start = 0
        for (bname, bdata) in self.bundles.items():
            bundles_indices[bname] = np.arange(
                current_start, current_start + bdata.shape[0])
            current_start += bdata.shape[0]

        assert current_start == nb_streamlines

        LOGGER.info("Fitting TSNE projection.")
        all_streamlines = np.concatenate(list(self.bundles.values()), axis=0)
        all_projected_streamlines = self.tsne.fit_transform(all_streamlines)

        if self.fig is None or self.ax is None:
            self.fig, self.ax = self._init_figure()

        self.ax.clear()

        for (bname, bdata) in self.bundles.items():
            bindices = bundles_indices[bname]
            proj_data = all_projected_streamlines[bindices]
            blabel = self.bundle_mapping.get(
                bname, bname) if self.bundle_mapping else bname

            self._plot_bundle(
                self.ax, proj_data[:, 0], proj_data[:, 1], blabel)

        self.ax.set_title("Best epoch {}".format(epoch))
        self._set_legend(self.ax, len(self.bundles))

        if self.prefix_numbering:
            filename = '{}_{}.png'.format(
                figure_name_prefix, epoch)
        else:
            filename = '{}.png'.format(figure_name_prefix)

        filename = os.path.join(self.save_dir, filename)
        self.fig.savefig(filename)

        self.current_plot_number += 1

    def _set_legend(self, ax, nb_bundles, order=True):
        if nb_bundles > 1:
            handles, labels = ax.get_legend_handles_labels()
            if order:
                labels, handles = zip(
                    *sorted(zip(labels, handles), key=lambda t: t[0]))
            ax.legend(handles, labels, fontsize=6,
                      loc='center left', bbox_to_anchor=(1, 0.5))

    def _plot_bundle(self, ax, dim1, dim2, blabel):
        ax.scatter(
            dim1,
            dim2,
            label=blabel,
            alpha=0.9,
            edgecolors='black',
            linewidths=0.5,
            color=self.bundle_color_manager.get_color(blabel)
        )

    def _init_figure(self):
        LOGGER.info("Init new figure for BundlesLatentSpaceVisualizer.")
        fig, ax = plt.subplots(1, 1)
        ax.set_title("Best epoch (?)")
        if self.fig_size is not None:
            fig.set_figwidth(self.fig_size[0])
            fig.set_figheight(self.fig_size[1])

        box_0 = ax.get_position()
        ax.set_position(
            [box_0.x0, box_0.y0, box_0.width * 0.8, box_0.height])

        return fig, ax

    def _to_numpy(self, data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        else:
            return data

    def _resample_max_subset_size(self, data: np.ndarray):
        """
        Resample the data to the max_subset_size.
        """
        _resampled = data
        if self.max_subset_size is not None:
            if not (self.max_subset_size > 0):
                raise ValueError(
                    "A max_subset_size of an integer value greater"
                    "than 0 is required.")

            # Only sample if we need to reduce the number of latent
            # streamlines to show on the plot.
            if (len(data) > self.max_subset_size):
                sample_indices = np.random.choice(
                    len(data),
                    self.max_subset_size, replace=False)

                _resampled = data[sample_indices]

        return _resampled
