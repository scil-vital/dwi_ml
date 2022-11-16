# -*- coding: utf-8 -*-
import logging

import numpy as np
import torch

from dwi_ml.data.processing.space.neighborhood import \
    prepare_neighborhood_vectors, extend_coordinates_with_neighborhood
from dwi_ml.data.processing.volume.interpolation import \
    interpolate_volume_in_neighborhood


def test_neighborhood_vectors():
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
    # Coords must be of shape (M, 3). Fails if coordinates are ints!
    current_coords = np.asarray([[10., 10., 10.], [3., 3., 3.]])
    m_coords = 2

    neighb_vec = prepare_neighborhood_vectors('grid', neighborhood_radius=1)
    neighb, _ = extend_coordinates_with_neighborhood(current_coords,
                                                     neighb_vec)
    assert len(neighb) == (26 + 1) * m_coords

    neighb_vec = prepare_neighborhood_vectors('axes', neighborhood_radius=0.5)
    neighb, _ = extend_coordinates_with_neighborhood(current_coords,
                                                     neighb_vec)
    assert len(neighb) == (6 + 1) * m_coords

    # Interpolating from fake data
    # values close to 3, 3, 3 should have values ~9 and values close to
    # 10, 10, 10 should have values ~30
    n_neigh = 6
    f_features = 2
    data = torch.tensor(np.random.rand(15, 15, 15, f_features))
    for i in range(15):
        for j in range(15):
            for k in range(15):
                data[i, j, k, 0] += i + j + k
                data[i, j, k, 1] += i + j + k

    # Without adding coordinates
    interpolated_data, _ = interpolate_volume_in_neighborhood(
        data, current_coords, neighb_vec)

    assert np.array_equal(interpolated_data.shape,
                          [m_coords, (n_neigh + 1) * f_features])
    assert 29 < np.mean(interpolated_data[0, :].numpy()) < 31
    assert 8 < np.mean(interpolated_data[1, :].numpy()) < 10

    # Adding coordinates
    interpolated_data, _ = interpolate_volume_in_neighborhood(
        data, current_coords, neighb_vec,
        add_vectors_to_data=True)

    assert np.array_equal(interpolated_data.shape,
                          [m_coords, (n_neigh + 1) * (f_features + 3)])


if __name__ == '__main__':
    logging.getLogger().setLevel(level='DEBUG')
    test_neighborhood_interpolation()
    test_neighborhood_vectors()
