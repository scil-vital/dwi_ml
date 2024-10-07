import math
from matplotlib.colors import ListedColormap
from matplotlib.cm import hsv
import numpy as np


def generate_dissimilar_color_map(nb_distinct_colors: int):
    """
    Create a colormap with a number of distinct colors.
    Needed to have bigger color maps for more bundles.

    Code directly copied from:
    https://stackoverflow.com/questions/42697933

    """
    if nb_distinct_colors == 0:
        nb_distinct_colors = 80

    nb_of_shades = 7
    nb_of_distinct_colors_with_mult_of_shades = int(
        math.ceil(nb_distinct_colors / nb_of_shades)
        * nb_of_shades)

    # Create an array with uniformly drawn floats taken from <0, 1)
    # partition
    linearly_distributed_nums = np.arange(
        nb_of_distinct_colors_with_mult_of_shades) / \
        nb_of_distinct_colors_with_mult_of_shades

    # We are going to reorganise monotonically growing numbers in such way
    # that there will be single array with saw-like pattern but each saw
    # tooth is slightly higher than the one before. First divide
    # linearly_distributed_nums into nb_of_shades sub-arrays containing
    # linearly distributed numbers.
    arr_by_shade_rows = linearly_distributed_nums.reshape(
        nb_of_shades, nb_of_distinct_colors_with_mult_of_shades //
        nb_of_shades)

    # Transpose the above matrix (columns become rows) - as a result each
    # row contains saw tooth with values slightly higher than row above
    arr_by_shade_columns = arr_by_shade_rows.T

    # Keep number of saw teeth for later
    number_of_partitions = arr_by_shade_columns.shape[0]

    # Flatten the above matrix - join each row into single array
    nums_distributed_like_rising_saw = arr_by_shade_columns.reshape(-1)

    # HSV colour map is cyclic we'll use this property
    # (https://matplotlib.org/tutorials/colors/colormaps.html#cyclic)
    initial_cm = hsv(nums_distributed_like_rising_saw)

    lower_partitions_half = number_of_partitions // 2
    upper_partitions_half = number_of_partitions - lower_partitions_half

    # Modify lower half in such way that colours towards beginning of
    # partition are darker .First colours are affected more, colours
    # closer to the middle are affected less
    lower_half = lower_partitions_half * nb_of_shades
    for i in range(3):
        initial_cm[0:lower_half, i] *= np.arange(0.2, 1, 0.8/lower_half)

    # Modify second half in such way that colours towards end of partition
    # are less intense and brighter. Colours closer to the middle are
    # affected less, colours closer to the end are affected more
    for i in range(3):
        for j in range(upper_partitions_half):
            modifier = np.ones(nb_of_shades) \
                - initial_cm[lower_half + j * nb_of_shades:
                             lower_half + (j + 1) * nb_of_shades, i]
            modifier = j * modifier / upper_partitions_half
            initial_cm[lower_half + j * nb_of_shades: lower_half +
                       (j + 1) * nb_of_shades, i] += modifier

    return ListedColormap(initial_cm)
