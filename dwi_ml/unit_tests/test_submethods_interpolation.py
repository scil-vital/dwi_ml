# -*- coding: utf-8 -*-
import logging

import numpy as np
import torch

from dwi_ml.data.processing.space.neighborhood import \
    prepare_neighborhood_vectors, extend_coordinates_with_neighborhood, \
    unflatten_neighborhood
from dwi_ml.data.processing.volume.interpolation import \
    interpolate_volume_in_neighborhood



nb_points = 2
nb_features = 2
fake_data = torch.zeros(15, 15, 15, nb_features)
for i in range(15):
    for j in range(15):
        for k in range(15):
            fake_data[i, j, k, 0] = i + j + k
            fake_data[i, j, k, 1] = i + j + k + 100


def test_neighborhood_interpolation():
    print("    -------------------")
    print("    Testing interpolation inside a neighborhood")

    fake_coords = torch.as_tensor([[10., 10., 10.],
                                   [3., 3., 3.]])
    fake_coords += torch.rand([2, 3])
    neighb_vec = prepare_neighborhood_vectors(
        'axes', neighborhood_radius=1, neighborhood_resolution=0.5)
    nb_neighb = len(neighb_vec)

    # Values at 10, 10, 10: ~[30, 130]
    # Values at 3, 3, 3: ~[9, 109]

    interpolated_data, _ = interpolate_volume_in_neighborhood(
        fake_data, fake_coords, neighb_vec)
    print("Interpolated data at each {} neighbor: \nvalues are feature1, "
          "feature2, at each neighbor (row) for each central coordinate (column)\n"
          "{}".format(nb_neighb, interpolated_data.shape, interpolated_data))

    assert np.array_equal(interpolated_data.shape,
                          [nb_points, nb_neighb * nb_features])

    # Coord 1: ~30 at each point + ~130 at each point = expected average of 80
    mean_coord1 = np.mean(interpolated_data[0, :].numpy())
    assert 80 < mean_coord1 < 83, \
        "Expecting mean interpolated value at coord 10 to be ~80, but got " \
        "{}".format(mean_coord1)


def test_neighborhood_interpolation_exact_data():
    print("    -------------------")
    print("    Testing interpolation inside a neighborhood: exact values")
    fake_coords = torch.as_tensor([[1., 5., 10.],  # --> Checking exact values
                                   [0., 0., 0.]])  # --> Checking outside image

    # Using resolution one, and round current coordinate, will be not be
    # interpolating anything = we can compare with raw (fake) data.
    neighb_vec = prepare_neighborhood_vectors('grid', neighborhood_radius=1,
                                              neighborhood_resolution=1)
    nb_neighb = len(neighb_vec)

    # Interpolating from fake data
    # Values at 1, 5, 10: ~[16, 116]
    # Values at 0, 0, 0: ~[0, 100], with unexisting values.

    interpolated_data, _ = interpolate_volume_in_neighborhood(
        fake_data, fake_coords, neighb_vec)
    print("Interpolated data at each {} neighbor (shape {}): \n"
          "values are feature1, feature2, at each neighbor (row) for each "
          "central coordinate (column)\n"
          "Visual supervision: there should be many zeros, for values with "
          "clipped data for coords outside the image."
          "{}".format(nb_neighb, interpolated_data.shape, interpolated_data))

    unflattened = unflatten_neighborhood(
        interpolated_data, neighb_vect=neighb_vec, neighb_rad=1, neighb_res=1,
        neighb_type='grid')
    assert np.array_equal(unflattened.shape, [nb_points, 3, 3, 3, nb_features])

    expected = fake_data[0:3, 4:7, 9:12, :]
    assert torch.equal(unflattened[0, :], expected), \
        "Expecting to interpolate the equivalent of initial data: \n" \
        "{}\n However, got: {}".format(expected, unflattened[0, :])


if __name__ == '__main__':
    logging.getLogger().setLevel(level='DEBUG')
    test_neighborhood_interpolation()
    test_neighborhood_interpolation_exact_data()
