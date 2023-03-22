# -*- coding: utf-8 -*-
import shutil
import sys
import os
from os.path import join, dirname, isfile, isdir

import subprocess

from scilpy.io.fetcher import get_home
from scilpy.io.utils import assert_outputs_exist, assert_inputs_exist

from dwi_ml.models.projects.transformer_visualisation_utils import \
    build_argparser_transformer_visu, tto_visualize_weights

# 1) Finding the jupyter notebook
dwi_ml_dir = dirname(dirname(__file__))
IPYNB_FILENAME = join(
    dwi_ml_dir, 'dwi_ml/models/projects/tto_visualize_weights.ipynb')

# 2) Jupyter notebook needs to know where to load the config file.
# Needs to be always at the same place.
# We choose to save it in our pytest dir.
home = get_home()
CONFIG_FILENAME = join(home, 'ipynb_tto_visualize_weights.config')


def main(argv):

    parser = build_argparser_transformer_visu()
    args = parser.parse_args()

    # Adding complete names with out_dir and prefix to args.
    if args.out_dir is None:
        args.out_dir = os.path.join(args.experiment_path, 'visu')
    args.out_html_file = os.path.join(
        args.out_dir, args.out_prefix + 'tto_visu.html')
    args.out_ipynb_file = os.path.join(
        args.out_dir, args.out_prefix + 'tto_visu.ipynb')
    args.out_config_file = os.path.join(
        args.out_dir, args.out_prefix + 'tto_visu.config')

    assert_inputs_exist(parser, [args.hdf5_file, args.input_streamlines],
                        args.reference)

    # Either run locally or launch for real
    if args.run_locally:
        # Will probably just print some HTML stuff in the terminal...
        tto_visualize_weights(args, parser)
    else:
        # 1. Test that we correctly found the ipynb file.
        if not isfile(IPYNB_FILENAME):
            raise ValueError(
                "We could not find the jupyter notebook file. Probably a "
                "coding error on our side. We expected it to be in {}"
                .format(IPYNB_FILENAME))

        # 2. Verify that output dir exists but not output files.
        if not isdir(args.out_dir):
            os.mkdir(args.out_dir)
        assert_outputs_exist(parser, args, [args.out_html_file,
                                            args.out_ipynb_file,
                                            args.out_config_file])

        # 3. Save the args to this script in a config file, to be read again
        # by the notebook in a new argparser instance.
        if isfile(CONFIG_FILENAME):  # In case a previous call failed.
            os.remove(CONFIG_FILENAME)
        with open(CONFIG_FILENAME, 'w') as f:
            f.write(' '.join(argv))

        # 4. Copy files locally
        shutil.copyfile(IPYNB_FILENAME, args.out_ipynb_file)
        shutil.copyfile(CONFIG_FILENAME, args.out_config_file)

        # 5. Launching.
        print("\n\n"
              "** We will by running the jupyter notebook and save its result "
              "as a html file: {} **\n\n".format(args.out_html_file))
        try:
            command = 'jupyter nbconvert --output-dir={:s} --execute {:s} ' \
                      '--to html'.format(args.out_html_file, IPYNB_FILENAME)
            print("Running command:\n\n>>{}\n\n".format(command))
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError:
            print("JUPYTER NOTEBOOK EXCEUTION WAS NOT SUCCESSFULL.")
            print("You may try to use --run_locally to debug.")
            print("You may try to run yourself the notebook copied in "
                  "--out_dir using the config file.")
            print("You may try to specify which kernel (i.e. which python "
                  "version) \nyour jupyter notebooks should run on:\n"
                  ">> (workon your favorite python environment)\n"
                  ">> pip install ipykernel\n"
                  ">> python -m ipykernel install --user\n")

        # 6. Delete config file.
        os.remove(CONFIG_FILENAME)


if __name__ == '__main__':
    main(sys.argv)
