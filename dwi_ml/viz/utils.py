from matplotlib.colors import ListedColormap
from skimage.color import hsv2rgb, rgb2lab, deltaE_ciede2000
import numpy as np


def generate_dissimilar_color_map(nb_distinct_colors: int):
    """
    Select `nb_distinct_colors` dissimilar colors by sampling HSV values and
    computing distances in the CIELAB space.

    Parameters:
        nb_distinct_colors (int): nb_distinct of colors to select.

    Returns:
        np.ndarray: Array of selected RGB colors.
    """

    # h_range (tuple): Range for the hue component.
    # s_range (tuple): Range for the saturation component.
    # v_range (tuple): Range for the value component.
    h_range = (0, 1)
    s_range = (0.8, 1)
    v_range = (0.8, 1)

    # Start with a random initial color
    rgb_colors = [[1, 0, 0]]
    while len(rgb_colors) < nb_distinct_colors:
        max_distance = -1
        best_color = None

        # Randomly generate a candidate color in HSV
        # Generate 100 candidates and pick the best one
        hue = np.random.uniform(h_range[0], h_range[1], 100)
        saturation = np.random.uniform(s_range[0], s_range[1], 100)
        value = np.random.uniform(v_range[0], v_range[1], 100)
        candidate_hsv = np.stack([hue, saturation, value], axis=1)
        candidate_rgb = hsv2rgb(candidate_hsv)

        # Compute the minimum distance to any selected color in LAB space
        distance = compute_cielab_distances(candidate_rgb, rgb_colors)
        distance = np.min(distance, axis=1)
        min_distance = np.max(distance)
        min_distance_id = np.argmax(distance)

        if min_distance > max_distance:
            max_distance = min_distance
            best_color = candidate_rgb[min_distance_id]

        rgb_colors = np.vstack([rgb_colors, best_color])

    return ListedColormap(rgb_colors)


def compute_cielab_distances(rgb_colors, compared_to=None):
    """
    Convert RGB colors to CIELAB and compute
    the Delta E (CIEDE2000) distance matrix.

    Parameters:
        rgb_colors (np.ndarray): Array of RGB colors.
        compared_to (np.ndarray): Array of RGB colors to compare against.
        If None, compare to rgb_colors.

    Returns:
        np.ndarray: nb_sample x nb_sample or \
                    nb_sample1 x nb_sample2 distance matrix.
    """
    # Convert RGB to CIELAB
    rgb_colors = np.clip(rgb_colors, 0, 1).astype(float)
    lab_colors_1 = rgb2lab(rgb_colors)

    if compared_to is None:
        lab_colors_2 = lab_colors_1
    else:
        compared_to = np.clip(compared_to, 0, 1).astype(float)
        lab_colors_2 = rgb2lab(compared_to)

    # Calculate the pairwise Delta E distances
    # using broadcasting and vectorization
    lab_colors_1 = lab_colors_1[:, np.newaxis, :]  # Shape (n1, 1, 3)
    lab_colors_2 = lab_colors_2[np.newaxis, :, :]  # Shape (1, n2, 3)

    # Vectorized Delta E calculation
    distance_matrix = deltaE_ciede2000(lab_colors_1, lab_colors_2,
                                       kL=1, kC=1, kH=1)

    return distance_matrix
