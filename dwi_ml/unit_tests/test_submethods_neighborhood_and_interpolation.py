# -*- coding: utf-8 -*-
import logging

import numpy as np
import torch

from dwi_ml.data.processing.space.neighborhood import \
    prepare_neighborhood_vectors, extend_coordinates_with_neighborhood, \
    unflatten_neighborhood
from dwi_ml.data.processing.volume.interpolation import \
    interpolate_volume_in_neighborhood


fake_coords = torch.as_tensor([[10., 10., 10.],
                              [3., 3., 3.]])
nb_points = 2
nb_features = 2
data = torch.zeros(15, 15, 15, nb_features)
for i in range(15):
    for j in range(15):
        for k in range(15):
            data[i, j, k, 0] = i + j + k
            data[i, j, k, 1] = i + j + k + 100


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


def test_neighborhood_interpolation_exact_data():
    print("    -------------------")
    print("    Testing interpolation inside a neighborhood: exact values")

    # Using radius one, will be not be interpolating anything.
    neighb_vec = prepare_neighborhood_vectors('grid', neighborhood_radius=1)
    nb_neighb = len(neighb_vec)

    # Interpolating from fake data
    print("Fake data shape: {}. \nValues at 10, 10, 10: {}\n"
          "Values at 3, 3, 3: {}. "
          .format(data.shape,  data[10, 10, 10], data[3, 3, 3]))

    interpolated_data, _ = interpolate_volume_in_neighborhood(
        data, fake_coords, neighb_vec)
    print("Interpolated data at each {} neighbor (shape {}): \n{}"
          .format(nb_neighb, interpolated_data.shape, interpolated_data))

    unflattened = unflatten_neighborhood(interpolated_data, neighb_vec)
    assert torch.equal(unflattened[0], data[9:11, 9:11, 9:11])
    assert torch.equal(unflattened[1], data[2:4, 2:4, 2:4])


def test_neighborhood_interpolation():
    print("    -------------------")
    print("    Testing interpolation inside a neighborhood")

    neighb_vec = prepare_neighborhood_vectors(
        'axes', neighborhood_radius=1, neighborhood_resolution=0.5)
    nb_neighb = 7

    # Interpolating from fake data: Adding
    # values close to 3, 3, 3 should have values ~9 and values close to
    # 10, 10, 10 should have values ~30
    print("Fake data shape: {}. \nValues at 10, 10, 10: {}\n"
          "Values at 3, 3, 3: {}. "
          .format(data.shape,  data[10, 10, 10], data[3, 3, 3]))

    interpolated_data, _ = interpolate_volume_in_neighborhood(
        data, fake_coords, neighb_vec)
    print("Interpolated data at each {}+1 neighbor (shape {}): \n{}"
          .format(nb_neighb, interpolated_data.shape, interpolated_data))

    assert np.array_equal(interpolated_data.shape,
                          [nb_points, nb_neighb * nb_features])
    assert 29 < np.mean(interpolated_data[0, :].numpy()) < 31
    assert 8 < np.mean(interpolated_data[1, :].numpy()) < 10


if __name__ == '__main__':
    logging.getLogger().setLevel(level='DEBUG')
    test_neighborhood_vectors()
    test_neighborhood_extended_coordinates()
    test_neighborhood_interpolation_exact_data()
    test_neighborhood_interpolation()
