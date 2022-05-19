# -*- coding: utf-8 -*-
import numpy as np

from dwi_ml.data.processing.space.neighborhood import \
    prepare_neighborhood_vectors, extend_coordinates_with_neighborhood


def test_neighborhood_points():
    npoints = prepare_neighborhood_vectors('axes', neighborhood_radius=1)
    assert len(npoints) == 6

    npoints = prepare_neighborhood_vectors('axes', neighborhood_radius=[1])
    assert len(npoints) == 6

    npoints = prepare_neighborhood_vectors('axes', neighborhood_radius=[1, 2])
    assert len(npoints) == 12

    npoints = prepare_neighborhood_vectors('grid', neighborhood_radius=1)
    assert len(npoints) == 26

    npoints = prepare_neighborhood_vectors('grid', neighborhood_radius=2)
    assert len(npoints) == 124


def test_neighborhood_interpolation():

    # Coords must be of shape (N, 3). Fails if coordinates are ints!
    current_coord = np.asarray([[10., 10., 10.]])

    npoints = prepare_neighborhood_vectors('grid', neighborhood_radius=1)
    neighb = extend_coordinates_with_neighborhood(current_coord, npoints)
    assert len(neighb) == 27

    npoints = prepare_neighborhood_vectors('axes', neighborhood_radius=0.42)
    neighb = extend_coordinates_with_neighborhood(current_coord, npoints)
    assert len(neighb) == 7


if __name__ == '__main__':
    test_neighborhood_interpolation()
    test_neighborhood_points()
