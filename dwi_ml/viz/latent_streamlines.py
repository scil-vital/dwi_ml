import os
import logging
from typing import Union, List, Tuple
from sklearn.manifold import TSNE
import numpy as np
import torch

import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)

class BundlesLatentSpaceVisualizer(object):
    """
    Utility class that wraps a t-SNE projection of the latent space for multiple bundles.
    The usage of this class is intented as follows:
        1. Create an instance of this class,
        2. Add the latent space streamlines for each bundle using "add_data_to_plot"
            with its corresponding label.
        3. Fit and plot the t-SNE projection using the "plot" method.
    
    t-SNE projection can only leverage the fit_transform() with all the data that needs to
    be projected at the same time since it aims to preserve the local structure of the data.
    """
    def __init__(self,
        save_dir: str = None,
        fig_size: Union[List, Tuple] = None,
        random_state: int = 42,
        max_subset_size: int = None,
        prefix_numbering: bool = False,
        reset_warning: bool = True
    ):
        """
        Parameters
        ----------
        save_path: str
            Path to save the figure. If not specified, the figure will be shown.
        fig_size: List[int] or Tuple[int]
            2-valued figure size (x, y)
        random_state: List
            Random state for t-SNE.
        max_subset_size: int
            In case of performance issues, you can limit the number of streamlines to plot
            for each bundle.
        prefix_numbering: bool
            If True, the saved figures will be numbered with the current plot number.
            The plot number is incremented after each call to the "plot" method.
        reset_warning: bool
            If True, a warning will be displayed when calling "plot" several times
            without calling "reset_data" in between to clear the data.
        """
        self.save_dir = save_dir
        
        # Make sure that self.save_dir is a directory and exists.
        if self.save_dir is not None:
            if not os.path.isdir(self.save_dir):
                raise ValueError("The save_dir should be a directory.")

        self.fig_size = fig_size
        self.random_state = random_state
        self.max_subset_size = max_subset_size
        self.prefix_numbering = prefix_numbering
        self.reset_warning = reset_warning
        
        self.current_plot_number = 0
        self.should_call_reset_before_plot = False

        self.tsne = TSNE(n_components=2, random_state=self.random_state)
        self.bundles = {}

    def reset_data(self):
        """
        Reset the data to plot. If you call plot several times without
        calling this method, the data will be accumulated.
        """
        # Not sure if resetting the TSNE object is necessary.
        self.tsne = TSNE(n_components=2, random_state=self.random_state)
        self.bundles = {}

    def add_data_to_plot(self, data: np.ndarray, label: str = '_'):
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
        if isinstance(data, torch.Tensor):
            latent_space_streamlines = data.detach().numpy()
        else:
            latent_space_streamlines = data
        
        if self.max_subset_size is not None:
            if not (self.max_subset_size > 0):
                raise ValueError("A max_subset_size of an integer value greater than 0 is required.")
            
            # Only sample if we need to reduce the number of latent streamlines
            # to show on the plot.
            if (len(latent_space_streamlines) > self.max_subset_size):
                sample_indices = np.random.choice(len(latent_space_streamlines), self.max_subset_size, replace=False)
                latent_space_streamlines = latent_space_streamlines[sample_indices] # (max_subset_size, 2)
        
        self.bundles[label] = latent_space_streamlines

    def plot(self, title: str = "", figure_name_prefix: str = 'lt_space'):
        """
        Fit and plot the t-SNE projection of the latent space streamlines.
        This should be called once after adding all the data to plot using "add_data_to_plot".
        
        Parameters
        ----------
        figure_name_prefix: str
            Name of the figure to be saved. This is just the prefix of the full file
            name as it will be suffixed with the current plot number if enabled.
        """
        if self.should_call_reset_before_plot and self.reset_warning:
            LOGGER.warning("You plotted another time without resetting the data. "
                           "The data will be accumulated, which might lead to unexpected results.")
            self.should_call_reset_before_plot = False
        elif not self.current_plot_number > 0:
            # Only enable the flag for the first plot.
            # So that the warning above is only displayed once.
            self.should_call_reset_before_plot = True

        nb_streamlines = sum(b.shape[0] for b in self.bundles.values())
        LOGGER.info("Plotting a total of {} streamlines".format(nb_streamlines))

        bundles_indices = {}
        current_start = 0
        for (bname, bdata) in self.bundles.items():
            bundles_indices[bname] = np.arange(current_start, current_start + bdata.shape[0])
            current_start += bdata.shape[0]

        assert current_start == nb_streamlines

        all_streamlines = np.concatenate(list(self.bundles.values()), axis=0)

        LOGGER.info("Fitting TSNE projection.")
        all_projected_streamlines = self.tsne.fit_transform(all_streamlines)

        LOGGER.info("New figure for t-SNE visualisation.")
        fig, ax = plt.subplots()
        ax.set_title(title)
        if self.fig_size is not None:
            fig.set_figheight(self.fig_size[0])
            fig.set_figwidth(self.fig_size[1])

        for (bname, bdata) in self.bundles.items():
            bindices = bundles_indices[bname]
            proj_data = all_projected_streamlines[bindices]
            ax.scatter(
                proj_data[:, 0],
                proj_data[:, 1],
                label=bname,
                alpha=0.9,
                edgecolors='black',
                linewidths=0.5,
            )
    
        if len(self.bundles) > 1:
            ax.legend()

        if self.save_dir is not None:
            filename = '{}_{}.png'.format(figure_name_prefix, self.current_plot_number) \
                if self.prefix_numbering else '{}.png'.format(figure_name_prefix)
            filename = os.path.join(self.save_dir, filename)
            fig.savefig(filename)
        else:
            plt.show()

        self.current_plot_number += 1

