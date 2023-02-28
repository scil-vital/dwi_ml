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


def resample_or_compress(sft, step_size, compress):
    if step_size is not None:
        # Note. No matter the chosen space, resampling is done in mm.
        logging.debug("            Resampling: {}".format(step_size))
        sft = resample_streamlines_step_size(sft, step_size=step_size)
    if compress is not None:
        logging.debug("            Compressing: {}".format(compress))
        sft = compress_sft(sft, compress)
    return sft
