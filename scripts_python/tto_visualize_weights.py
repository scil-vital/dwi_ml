# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname

from scilpy.io.fetcher import get_home
from scilpy.io.utils import assert_inputs_exist

from dwi_ml.testing.projects.transformer_visualisation import \
    visualize_weights_using_jupyter, tto_visualize_weights
from dwi_ml.testing.projects.transformer_visualisation_utils import \
    build_argparser_transformer_visu


def main(argv):

    parser = build_argparser_transformer_visu()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.hdf5_file, args.input_streamlines],
                        args.reference)

    # Either run locally or launch for real
    if args.run_locally:
        # Will probably just print some HTML stuff in the terminal...
        tto_visualize_weights(args, parser)
    else:
        # 1) Finding the jupyter notebook
        dwi_ml_dir = dirname(dirname(__file__))
        ipynb_filename = join(
            dwi_ml_dir, 'dwi_ml/visu/projects/tto_visualize_weights.ipynb')

        # 2) Jupyter notebook needs to know where to load the config file.
        # Needs to be always at the same place.
        # We choose to save it in our pytest dir.
        home = get_home()
        config_filename = join(home, 'ipynb_tto_visualize_weights.config')

        visualize_weights_using_jupyter(ipynb_filename, config_filename,
                                        parser, args, argv)


if __name__ == '__main__':
    main(sys.argv)
