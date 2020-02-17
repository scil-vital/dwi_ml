
import torch

from scil_vital.shared.code.transformation.space import convert_mm2vox


def torch_trilinear_interpolation(volume: torch.Tensor, coords: torch.Tensor) \
        -> torch.Tensor:
    """Evaluates the data volume at given coordinates using trilinear
    interpolation on a torch tensor.

    Interpolation is done using the device on which the volume is stored.

    * Note. There is a function in torch:
    torch.nn.functional.interpolation with mode trilinear
    But it resamples volumes, not coordinates.

    Parameters
    ----------
    volume : torch.Tensor with 3D or 4D shape
        The input volume to interpolate from
    coords : torch.Tensor with shape (N,3)
        The coordinates where to interpolate

    Returns
    -------
    output : torch.Tensor with shape (N, #modalities)
        The list of interpolated values

    References
    ----------
    [1] https://spie.org/samples/PM159.pdf
    """
    # Get device, and make sure volume and coords are using the same one
    assert volume.device == coords.device,\
        "volume on device: {}; " \
        "coords on device: {}".format(volume.device, coords.device)
    device = volume.device

    # Send data to device
    idx_torch = torch.as_tensor(idx, dtype=torch.float, device=device)
    B1_torch = torch.as_tensor(B1, dtype=torch.float, device=device)

    if volume.dim() <= 2 or volume.dim() >= 5:
        raise ValueError("Volume must be 3D or 4D!")

    if volume.dim() == 3:
        # torch needs indices to be cast to long
        indices_unclipped = \
            (coords[:, None, :] + idx_torch).reshape((-1, 3)).long()

        # Clip indices to make sure we don't go out-of-bounds
        lower = torch.as_tensor([0, 0, 0]).to(device)
        upper = (torch.as_tensor(volume.shape) - 1).to(device)
        indices = torch.min(torch.max(indices_unclipped, lower), upper)

        # Fetch volume data at indices
        p = volume[indices[:, 0], indices[:, 1], indices[:, 2]]
        p = p.reshape((coords.shape[0], -1)).t()

        d = coords - torch.floor(coords)
        dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
        Q1 = torch.stack([torch.ones_like(dx), dx, dy, dz,
                          dx * dy, dy * dz, dx * dz,
                          dx * dy * dz], dim=0)
        output = torch.sum(p * torch.mm(B1_torch.t(), Q1), dim=0)

        return output

    if volume.dim() == 4:
        # 8 coordinates of the corners of the cube, for each input coordinate
        indices_unclipped = \
            torch.floor(coords[:, None, :] + idx_torch).reshape((-1, 3)).long()

        # Clip indices to make sure we don't go out-of-bounds
        lower = torch.as_tensor([0, 0, 0], device=device)
        upper = torch.as_tensor(volume.shape[:3], device=device) - 1
        indices = torch.min(torch.max(indices_unclipped, lower), upper)

        # Fetch volume data at indices
        p = volume[indices[:, 0], indices[:, 1], indices[:, 2], :]
        p = p.reshape((coords.shape[0], 8, volume.shape[-1]))

        d = coords - torch.floor(coords)
        dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
        Q1 = torch.stack([torch.ones_like(dx), dx, dy, dz,
                          dx * dy, dy * dz, dx * dz,
                          dx * dy * dz], dim=0)
        output = \
            torch.sum(p * torch.mm(B1_torch.t(), Q1).t()[:, :, None], dim=1)

        return output

    raise ValueError("There was a problem with the volume's number of "
                     "dimensions!")

