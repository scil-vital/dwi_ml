import logging

from scilpy.tracking.tools import resample_streamlines_step_size
from scilpy.utils.streamlines import compress_sft


def add_resample_or_compress_arg(p):
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        '--step_size', type=float, metavar='s',
        help="Step size to resample the data (in mm). Default: None")
    g.add_argument(
        '--compress', type=float, metavar='r', const=0.01, nargs='?',
        help="Compression ratio. Default: None. Default if set: 0.01.\n"
             "If neither step_size nor compress are chosen, streamlines "
             "will be kept \nas they are.")
