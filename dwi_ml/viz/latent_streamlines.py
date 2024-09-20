import logging

from typing import Union, List, Tuple
from sklearn.manifold import TSNE
import numpy as np
import torch

import matplotlib.pyplot as plt

def plot_latent_streamlines(
        encoded_streamlines: Union[np.ndarray, torch.Tensor],
        save_path: str = None,
        fig_size: Union[List, Tuple] = None,
        random_state: int = 42,
        max_subset_size: int = None
    ):
    """
    Projects and plots the latent space representation
    of the streamlines using t-SNE dimensionality reduction.

    Parameters
    ----------
    encoded_streamlines: Union[np.ndarray, torch.Tensor]
        Latent space streamlines to plot of shape (N, latent_space_dim).
    save_path: str
        Path to save the figure. If not specified, the figure will be shown.
    fig_size: List[int] or Tuple[int]
        2-valued figure size (x, y)
    random_state: int
        Random state for t-SNE.
    max_subset_size: int:
        In case of performance issues, you can limit the number of streamlines to plot.
    """
    
    if isinstance(encoded_streamlines, torch.Tensor):
        latent_space_streamlines = encoded_streamlines.cpu().numpy()
    else:
        latent_space_streamlines = encoded_streamlines
    
    if max_subset_size is not None:
        if not (max_subset_size > 0):
            raise ValueError("A max_subset_size of an integer value greater than 0 is required.")
        
        # Only sample if we need to reduce the number of latent streamlines
        # to show on the plot.
        if (len(latent_space_streamlines) > max_subset_size):
            sample_indices = np.random.choice(len(latent_space_streamlines), max_subset_size, replace=False)
            latent_space_streamlines = latent_space_streamlines[sample_indices] # (max_subset_size, 2)

    # Project the data into 2 dimensions.
    tsne = TSNE(n_components=2, random_state=random_state)
    X_tsne = tsne.fit_transform(latent_space_streamlines) # Output (N, 2)


    logging.info("New figure for t-SNE visualisation.")
    fig, ax = plt.subplots()
    if fig_size is not None:
        fig.set_figheight(fig_size[0])
        fig.set_figwidth(fig_size[1])

    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.9, edgecolors='black', linewidths=0.5)
    
    if save_path is not None:
        fig.savefig(save_path)
    else:
        plt.show()


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
        save_path: str = None,
        fig_size: Union[List, Tuple] = None,
        random_state: int = 42,
        max_subset_size: int = None
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
        max_subset_size:
            In case of performance issues, you can limit the number of streamlines to plot
            for each bundle.
        """
        self.save_path = save_path
        self.fig_size = fig_size
        self.random_state = random_state
        self.max_subset_size = max_subset_size

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
            latent_space_streamlines = data.cpu().numpy()
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

    def plot(self):
        """
        Fit and plot the t-SNE projection of the latent space streamlines.
        This should be called once after adding all the data to plot using "add_data_to_plot".
        """
        nb_streamlines = sum(b.shape[0] for b in self.bundles.values())
        logging.info("Plotting a total of {} streamlines".format(nb_streamlines))

        bundles_indices = {}
        current_start = 0
        for (bname, bdata) in self.bundles.items():
            bundles_indices[bname] = np.arange(current_start, current_start + bdata.shape[0])
            current_start += bdata.shape[0]

        assert current_start == nb_streamlines

        all_streamlines = np.concatenate(list(self.bundles.values()), axis=0)

        logging.info("Fitting TSNE projection.")
        all_projected_streamlines = self.tsne.fit_transform(all_streamlines)

        logging.info("New figure for t-SNE visualisation.")
        fig, ax = plt.subplots()
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
    
        ax.legend()

        if self.save_path is not None:
            fig.savefig(self.save_path)
        else:
            plt.show()

