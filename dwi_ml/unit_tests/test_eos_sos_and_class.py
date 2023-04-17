#!/usr/bin/env python
import torch
from dipy.data import get_sphere
from dwi_ml.data.processing.streamlines.sos_eos_management import \
    add_label_as_last_dim, add_zeros_sos_eos, convert_dirs_to_class
from dwi_ml.data.spheres import TorchSphere
from matplotlib import pyplot as plt

streamline = torch.as_tensor([[1.0, 1, 1],
                              [2, 1, 1]])
batch = [streamline]


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
    idx_sos = 100
    idx_eos = 101

    # Expected values:
    true_idx = torch_sphere.find_closest(streamline)
    true_one_hot = torch.zeros((4, 102))
    true_one_hot[0, idx_sos] = 1
    true_one_hot[1, true_idx[0]] = 1
    true_one_hot[2, true_idx[1]] = 1
    true_one_hot[3, idx_eos] = 1

    # 1. No smoothing, to index, SOS, EOS
    streamline_as_idx = convert_dirs_to_class(
        batch, torch_sphere, smooth_labels=False, add_sos=True, add_eos=True,
        to_one_hot=False)[0]
    assert torch.equal(
        streamline_as_idx,
        torch.as_tensor([idx_sos, true_idx[0], true_idx[1], idx_eos]))

    # 2. No smoothing, to one-hot, SOS, EOS
    streamline_as_one_hot = convert_dirs_to_class(
        batch, torch_sphere, smooth_labels=False, add_sos=True, add_eos=True,
        to_one_hot=True)[0]
    streamline_as_idx = torch.argmax(streamline_as_one_hot, dim=1)
    assert torch.equal(
        streamline_as_idx,
        torch.as_tensor([idx_sos, true_idx[0], true_idx[1], idx_eos]))
    assert torch.equal(streamline_as_one_hot, true_one_hot)

    # 3. Smoothing, to one-hot, SOS, EOS
    streamline_as_one_hot = convert_dirs_to_class(
        batch, torch_sphere, smooth_labels=True, add_sos=True, add_eos=True,
        to_one_hot=True)[0]
    print(streamline_as_one_hot.sum(dim=1))
    streamline_as_idx = torch.argmax(streamline_as_one_hot, dim=1)
    assert torch.equal(
        streamline_as_idx,
        torch.as_tensor([idx_sos, true_idx[0], true_idx[1], idx_eos]))

    if plot_sphere:
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(sphere.x, sphere.y, sphere.z,
                   c=streamline_as_one_hot[1, :-2].numpy())

        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(sphere.x, sphere.y, sphere.z,
                   c=streamline_as_one_hot[2, :-2].numpy())
        plt.show()


if __name__ == '__main__':
    test_labels()
    test_zeros()
    test_classes(plot_sphere=True)
