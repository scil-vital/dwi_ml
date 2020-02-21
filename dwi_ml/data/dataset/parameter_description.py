PARAMETER_DESCRIPTION = """
    path : str
        Path to the processed .hdf5 file.
    rng : np.random.RandomState
        Random number generator.
    name : str
        Name of the dataset. If none, use the basename of the given .hdf5
        file. [None]
    use_streamline_noise : float
        If set, add random gaussian noise to streamline coordinates
        on-the-fly. Noise variance is 0.1 * step-size, or 0.1mm if no step
        size is used. [False]
    step_size : float
        Constant step size that every streamline should have between points
        (in mm). If None, train on streamlines as they are (ex, compressed).
        Note that you probably already fixed a step size when creating your
        dataset, but you could use a different one here if you wish. [None]
    add_neighborhood : float
        Add neighborhood points at the given distance (in mm) in each
        direction (nb_neighborhood_axes). [None] (None and 0 have the same
        effect).
    nb_neighborhood_axes: int
        Nb of axes (directions) to get the neighborhood voxels. This is only
        used if do_interpolation is True. Currently, 6 is the default and
        only implemented version (left, right, front, behind, up, down).
                                                                                                #ToDO. Raises notImplemented if not 6.
    streamlines_cut_ratio : float
        Percentage of streamlines to randomly cut in each batch. The reason
        for cutting is to help the ML algorithm to track from the middle of
        WM by having already seen half-streamlines. If you are using
        interface seeding, this is not necessary. If None, do not split
        streamlines. [None]
    add_previous_dir : bool
        If set, concatenate the previous streamline direction as input.
        [False]
    do_interpolation : bool
        If True, do the interpolation in the collate_fn (worker function).
        In this case, collate_fn returns PackedSequences ready for the
        model. [False]
    device : torch.device
        Device on which to process data. ['cpu']
    taskman_managed : bool
        If True, taskman manages the experiment. Do not output progress bars
        and instead output special messages for taskman. [False]
    """