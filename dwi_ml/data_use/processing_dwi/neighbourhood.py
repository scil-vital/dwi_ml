def get_interp_neighborhood_vectors(radius: float, nb_axes: int = 6,
                                    convert_mm_to_vox: bool = False,
                                    affine: np.ndarray = None) -> np.ndarray:
    """Returns predefined neighborhood directions at exactly `radius` length
       For now: Use the 6 main axes as neighbors directions, plus (0,0,0) to
       keep current position.

    Parameters
    ----------
    radius : float                                                                                          # ToDo. Possibilité de prendre des "shells" de direction, un peu pour imiter
                                                                                                            # de prendre "tous les voxels à l'intérieur d'un certain radius". ça veut rien dire
                                                                                                            # avec de l'interpolation mais on pourrait vouloir les voxels à 1mm, 2mm, etc.
                                                                                                            # donc radius = float ou list[float]
        Distance to neighbors.
    nb_axes: int
        Nb of axes around the center. Default (6) is up, down, left, right,
        behind, in front.
    convert_mm_to_vox: bool
        If true, will convert noise from mm to vox_iso using affine first.
        Note that we don't support vox to mm yet, nor mm to vox_noniso.
        [False]
    affine: np.ndarray
        Needed if convert_noise_space is True. Ex : affine_vox2rasmm


    Returns
    -------
    directions : np.ndarray with shape (n_directions, 3)

    Notes
    -----
    Coordinates are in voxel-space.
    """

    # Dealing with spaces:
    if convert_mm_to_vox:
        radius = convert_mm2vox(radius, affine)

    if nb_axes == 6:
        tmp_axes = np.identity(3)
        axes = np.concatenate((tmp_axes, -tmp_axes))
    else:
        raise NotImplementedError                                                                                   # à faire, Emma. Ex: Interp nb_axes sur une sphère.

    axes = np.concatenate(([[0, 0, 0]], axes))
    neighborhood_vectors = axes * radius

    return neighborhood_vectors


def extend_coords_with_interp_neighborhood_vectors(list_points,
                                                   neighborhood_vectors):
    """
    Get the interpolated neighborhood coordinates for each point in a list of
    (x,y,z) points.

    Params
    ------
    list_coords: ?
        A list of points as (x,y,z) coordinates

    Returns
    -------

    """
    n_input_points = list_points.shape[0]

    # 1. We have a list of coordinates: [point1 point2 ...]

    # 2. We repeat them.   [p1 p1... p2 p2 ... ...]
    list_points = np.repeat(list_points, neighborhood_vectors.shape[0], axis=0)

    # 3. Ex, if vectors = [middle up down left right],
    #    we tile, i.e. [0 u d l r , 0 u d l r, ...]
    tiled_v = np.tile(neighborhood_vectors, (n_input_points, 1))

    # 4. We add 2 and 3:  [p1 p1+left p1+right, ...]
    list_points[:, :3] += tiled_v

    return list_points