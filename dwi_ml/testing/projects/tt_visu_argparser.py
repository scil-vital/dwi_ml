# -*- coding: utf-8 -*-

"""
TT weights visualisation choices.

Output options. Choose any number (at least one).
     ** Currently, options bertviz and as_matrices use only the first
     streamline in the data.

    1) 'as_matrices': Show attention as matrices.
       If bertviz is also chosen, matrices will show in the html.
    2) 'bertviz': Show using bertviz head_view visualisation.
       Will create a html file that can be viewed (see --out_dir)
    3) 'color_sft': Save a colored sft. Streamlines are duplicated at all
       lengths. Color of the streamline of length n is the weight of each point
       when getting the next direction at point n.
    4) 'color_x_y': Save two colored sft:
        - Projection on x: This is a measure of the importance of each
        point on the streamline.
        - Projection on y: This is an indication of where we were looking
        at when deciding the next direction at each point on the
        streamline.
        The projection technique depends on the chosen rescaling options.
    5) 'bertviz_locally': Run the bertviz without using jupyter.
       (Debugging purposes. Output will not show, but html stuff will print
       in the terminal.)
"""
import argparse

from scilpy.io.utils import (add_overwrite_arg, add_reference_arg)

from dwi_ml.io_utils import (add_arg_existing_experiment_path,
                             add_logging_arg, add_memory_args)
from dwi_ml.testing.utils import add_args_testing_subj_hdf5


def build_argparser_transformer_visu():
    """
    This needs to be in a module, to be imported in the jupyter notebook. Do
    not put in the script file.
    """
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    # -------------
    # Positional args: Args to load data
    # -------------
    add_arg_existing_experiment_path(p)
    add_args_testing_subj_hdf5(p, ask_input_group=True)
    p.add_argument('in_sft',
                   help="A small tractogram; a bundle of streamlines that "
                        "should be \nuniformized. Else, see option "
                        "--align_endpoints")

    # --------------
    # Args to load data
    # --------------
    g = p.add_argument_group("Loading options")
    g.add_argument('--uniformize_endpoints', action='store_true',
                   help="If set, try aligning endpoints of the sft. Will use "
                        "the automatic \nalignment. For more options, align "
                        "your streamlines first, using \n"
                        "  >> scil_tractogram_uniformize_endpoints.py.\n")
    g.add_argument('--reverse_lines', action='store_true',
                   help="If set, reverses all streamlines first.\n"
                        "(With option --uniformize_endpoints, reversing is "
                        "done after.)")

    # --------------
    # Args to save data
    # --------------
    g = p.add_argument_group("Visualization options. At least one required.")
    g.add_argument('--as_matrices', action='store_true',
                   help="See description above.")
    g.add_argument('--bertviz', action='store_true',
                   help="See description above.")
    g.add_argument('--color_sft', action='store_true',
                   help="See description above.")
    g.add_argument('--color_x_y', action='store_true',
                   help="See description above.")
    g.add_argument('--bertviz_locally',
                   help="See description above.")

    g = p.add_argument_group("Saving options")
    g.add_argument(
        '--out_prefix', metavar='name', default='tt',
        help="Prefix of the all output files. Do not include a path."
             "Default: 'tt'\n"
             "Suffixes are: (here, example, for the encoder matrix)\n"
             "   1) as_matrices: _matrix_encoder.png\n"
             "   2) bertviz: _bertviz.html, _bertviz.ipynb, _bertviz.config\n"
             "   3) color_sft: _encoder_colored_sft_*.trk\n"
             "      (with layer and head information)\n"
             "   4) color_x_y: _encoder_colored_sft_importance_*.trk and \n"
             "                   _encoder_colored_sft_importance_where_looked.trk\n"
             "   5) bertviz_locally: None")
    g.add_argument(
        '--out_dir', metavar='d',
        help="Output directory where to save the output files.\n"
             "Default: experiment_path/visu_weights")

    # --------------
    # Options
    # --------------
    g = p.add_argument_group(
        "Weights processing options",
        description="By default, the sum per row in the matrix is 1. "
                    "Meaning: If all positions are equally important, their "
                    "value will be 1/n in the nth row.")
    gg = g.add_mutually_exclusive_group()
    gg.add_argument('--rescale_0_1', action='store_true',
                    help="If set, rescale to max 1 per row. X = X/max(row)")
    gg.add_argument('--rescale_z', action='store_true',
                    help="If set, transform each value to X = (X - mu) / std, "
                         "where \nmu and std are computed per row.")
    gg.add_argument('--rescale_non_lin', action='store_true',
                    help="If set, transform each value so that values below "
                         "the equal \nattention are transformed to [0-0.5], "
                         "and values above to [0.5, 1].\n"
                         "(Ex: At point #3, 0.33 --> 0.5. At point #40, "
                         "0.025 --> 0.5.")

    g = p.add_argument_group("Options defining how to deal with the heads")
    gg = g.add_mutually_exclusive_group()
    gg.add_argument('--group_heads', action='store_true',
                    help="If true, average all heads (per layer, per "
                         "attention type).\n"
                         "To regroup using maximum instead, use "
                         "--group_with_max")
    gg.add_argument('--group_all', action='store_true',
                    help="If true, average all heads in all layers (per "
                         "attention type).\n"
                         "To regroup using maximum instead, use "
                         "--group_with_max")
    g.add_argument('--group_with_max', action='store_true',
                   help="Default grouping option is to average heads. Use "
                        "this option to group \nhead using their maximal "
                        "values.\n"
                        "NOTE: Average is done BEFORE rescaling (averaging "
                        "the raw weight),\nand max is done AFTER rescaling "
                        "(max of the rank use usefullness).")

    g = p.add_argument_group("Matrices options")
    g.add_argument('--show_now', action='store_true',
                   help="If set, shows the matrices on screen. Else, only "
                        "saves them.")
    g.add_argument('--resample_plots', type=int, metavar='nb',
                   dest='resample_attention',
                   help="Streamlines will be sampled (nb points) as decided "
                        "by the model. \nHowever, attention shown as a matrix "
                        "can be resampled.")

    g = add_memory_args(p)
    g.add_argument('--batch_size', type=int, metavar='n',
                   help="Batch size in number of streamlines. If not set, "
                        "uses all streamlines \nin one batch.")
    add_reference_arg(p)
    add_logging_arg(p)
    add_overwrite_arg(p)

    return p
