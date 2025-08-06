# -*- coding: utf-8 -*-
import logging

import torch

from dwi_ml.data.processing.space.neighborhood import \
    prepare_neighborhood_vectors, extend_coordinates_with_neighborhood


fake_coords = torch.as_tensor([[10., 10., 10.],
                              [3., 3., 3.]])
nb_points = 2
nb_features = 2


def test_neighborhood_vectors():
    print("    -------------------")
    print("    Testing the number of points per type of neighborhood")

    npoints = prepare_neighborhood_vectors('axes', neighborhood_radius=1)
    assert len(npoints) == 7

    npoints = prepare_neighborhood_vectors('axes', neighborhood_radius=2)
    assert len(npoints) == 13

    npoints = prepare_neighborhood_vectors('grid', neighborhood_radius=1)
    assert len(npoints) == 27

    npoints = prepare_neighborhood_vectors('grid', neighborhood_radius=2)
    assert len(npoints) == 125

    print("OK.")


def test_neighborhood_extended_coordinates():
    print("    -------------------")
    print("    Testing extending coordinates with neighborhood vectors")

    # Coords must be of shape (M, 3). Fails if coordinates are ints!
    neighb_vec = prepare_neighborhood_vectors(
        'grid', neighborhood_radius=1, neighborhood_resolution=0.5)
    neighb, _ = extend_coordinates_with_neighborhood(fake_coords, neighb_vec)
    assert len(neighb) == 27 * nb_points

    neighb_vec = prepare_neighborhood_vectors(
        'axes', neighborhood_radius=1, neighborhood_resolution=0.5)
    neighb, _ = extend_coordinates_with_neighborhood(fake_coords, neighb_vec)
    assert len(neighb) == 7 * nb_points

    print("OK.")


if __name__ == '__main__':
    logging.getLogger().setLevel(level='DEBUG')
    test_neighborhood_vectors()
    test_neighborhood_extended_coordinates()
