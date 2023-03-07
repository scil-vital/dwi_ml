#!/usr/bin/env python
import torch
from dipy.data import get_sphere
from dwi_ml.data.processing.streamlines.sos_eos_management import \
    add_label_as_last_dim, add_zeros_sos_eos, convert_dirs_to_class
from dwi_ml.data.spheres import TorchSphere
from matplotlib import pyplot as plt

two_points = torch.as_tensor([[1.0, 1, 1],
                              [2, 1, 1]])
batch = [two_points]


def test_labels():
    updated = add_label_as_last_dim(batch, add_sos=True, add_eos=True)
    assert torch.equal(updated[0], torch.as_tensor([[0, 0, 0, 1, 0.0],
                                                    [1, 1, 1, 0, 0],
                                                    [2, 1, 1, 0, 0],
                                                    [0, 0, 0, 0, 1]]))


def test_zeros():
    updated = add_zeros_sos_eos(batch, add_sos=True, add_eos=True)
    assert torch.equal(updated[0], torch.as_tensor([[0, 0, 0],
                                                    [1, 1, 1],
                                                    [2, 1, 1],
                                                    [0, 0, 0]]))


def test_classes(plot_sphere=False):
    sphere = get_sphere('repulsion100')  # Sphere
    torch_sphere = TorchSphere(sphere)

    # 1. No smoothing, to index
    updated = convert_dirs_to_class(batch, torch_sphere, smooth_labels=False,
                                    add_sos=True, add_eos=True,
                                    to_one_hot=False)

    idx = torch_sphere.find_closest(two_points)
    assert torch.equal(updated[0], torch.as_tensor([100, idx[0], idx[1], 101]))

    # 2. No smoothing, to one-hot
    updated = convert_dirs_to_class(batch, torch_sphere, smooth_labels=False,
                                    add_sos=True, add_eos=True,
                                    to_one_hot=True)

    idx = torch_sphere.find_closest(two_points)
    assert idx[0] == idx[1]

    one_hot = torch.zeros((4, 102))
    one_hot[0, 100] = 1
    one_hot[1, idx[0]] = 1
    one_hot[2, idx[1]] = 1
    one_hot[-1, 101] = 1
    assert torch.equal(updated[0], one_hot)

    # 3. Smoothing
    updated = convert_dirs_to_class(batch, torch_sphere, smooth_labels=True,
                                    add_sos=True, add_eos=True,
                                    to_one_hot=True)

    assert updated[0][0, 100] == 1
    assert updated[0][1, 100] == 0
    assert updated[0][2, 100] == 0
    assert updated[0][3, 101] == 1

    if plot_sphere:
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(sphere.x, sphere.y, sphere.z, c=updated[0][1, :-2].numpy())

        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(sphere.x, sphere.y, sphere.z, c=updated[0][2, :-2].numpy())
        plt.show()


if __name__ == '__main__':
    test_labels()
    test_zeros()
    test_classes(plot_sphere=True)
