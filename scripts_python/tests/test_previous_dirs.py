#!/usr/bin/env python
import torch
from dwi_ml.data.processing.streamlines.post_processing import \
    compute_n_previous_dirs


def main():
    # Short streamline
    streamline1 = torch.tensor([[3.2, 31.2, 45.7]], dtype=torch.float32)

    # Short streamline
    streamline2 = torch.tensor([[5, 5, 5],
                                [6, 7, 8]], dtype=torch.float32)

    # Long streamline
    sub_streamline = torch.tensor([[1, 1, 1],
                                   [2, 2, 2],
                                   [6, 6, 6],
                                   [4, 4, 4]], dtype=torch.float32)
    streamline3 = torch.cat((sub_streamline, sub_streamline, sub_streamline))

    empty_coord = torch.zeros((1, 3), dtype=torch.float32)
    for streamlines in [streamline1, streamline2, streamline3]:
        print("TEST ON A STREAMLINE OF LENGTH {}".format(len(streamlines)))
        streamlines = [streamlines]

        print("Streamlines:\n{}".format(streamlines))
        streamline_dirs = [torch.as_tensor(s[1:] - s[:-1],
                                           dtype=torch.float32)
                           for s in streamlines]
        print("Streamlines dirs:\n{}".format(streamline_dirs))

        print("Four previous dirs")
        test = compute_n_previous_dirs(streamline_dirs, 4, empty_coord)
        print(test)

        print("\n\n")


if __name__ == '__main__':
    main()
