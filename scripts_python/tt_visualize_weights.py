# -*- coding: utf-8 -*-
import os
import shutil
import subprocess
import sys
from os.path import dirname

from scilpy.io.utils import (assert_inputs_exist, assert_outputs_exist)

from dwi_ml.testing.projects.transformer_visualisation import (
    build_argparser_transformer_visu, get_config_filename,
    tt_visualize_weights)


# Note. To use through jupyter, the file
# must also be up-to-date.
# To modify it as developers:
# jupyter notebook dwi_ml/testing/projects/tt_visualise_weights.ipynb
# Do not run it there though! Else, saving it will save the outputs too!


def main():
    # Getting the raw args. Will be sent to jupyter.
    argv = sys.argv

    parser = build_argparser_transformer_visu()
    args = parser.parse_args()

    # Doing some verification now. Will be done again through jupyter but it
    # may help user to verify now.
    assert_inputs_exist(parser, [args.hdf5_file, args.input_streamlines],
                        args.reference)
    if not os.path.isdir(args.experiment_path):
        parser.error("Experiment {} not found.".format(args.experiment_path))

    # Either run locally or launch for real
    if args.run_locally:
        # Will probably just print some HTML stuff in the terminal...
        print("--DEBUGGING MODE--\n"
              "We will run the script but it will not save any output!")
        tt_visualize_weights(args, parser)
    else:
        # 1) Finding the jupyter notebook
        dwi_ml_dir = dirname(dirname(__file__))
        ipynb_filename = os.path.join(
            dwi_ml_dir, 'dwi_ml/testing/projects/tt_visualize_weights.ipynb')
        if not os.path.isfile(ipynb_filename):
            raise ValueError(
                "We could not find the jupyter notebook file. Probably a "
                "coding error on our side. We expected it to be in {}"
                .format(ipynb_filename))

        # 2) Verify that output dir exists but not output files.
        if args.out_dir is None:
            args.out_dir = os.path.join(args.experiment_path, 'visu')
        if not os.path.isdir(args.out_dir):
            os.mkdir(args.out_dir)
        out_html_filename = args.out_prefix + 'tt_visu.html'
        out_html_file = os.path.join(args.out_dir, out_html_filename)
        out_ipynb_file = os.path.join(
            args.out_dir, args.out_prefix + 'tt_visu.ipynb')
        out_config_file = os.path.join(
            args.out_dir, args.out_prefix + 'tt_visu.config')
        assert_outputs_exist(parser, args,
                             [out_html_file, out_ipynb_file, out_config_file])

        # 2) Save the args to this script in a config file, to be read again
        # by the notebook in a new argparse instance.
        # Jupyter notebook needs to know where to load the config file.
        # Needs to be always at the same place because we cannot send an
        # argument to jupyter. Or, we could ask it here and tell user to
        # add it manually inside the jupyter notebook... complicated.
        config_filename = get_config_filename()
        if os.path.isfile(config_filename):  # In case a previous call failed.
            os.remove(config_filename)
        with open(config_filename, 'w') as f:
            f.write(' '.join(argv))

        # 4. Copy files locally so that user can see it.
        shutil.copyfile(ipynb_filename, out_ipynb_file)
        shutil.copyfile(config_filename, out_config_file)

        # 5. Launching.
        print("\n\n"
              "********\n"
              "* We will run the script through jupyter notebook! \n"
              "* We will save the results as a html file:\n"
              "* {}\n"
              "********\n\n".format(out_html_file))
        try:
            command = 'jupyter nbconvert --output-dir={:s} --output={} ' \
                      '--execute {:s} --to html' \
                .format(args.out_dir, out_html_filename, ipynb_filename)
            print("Running command:\n\n>>{}\n\n".format(command))
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError:
            print("JUPYTER NOTEBOOK EXCEUTION WAS NOT SUCCESSFULL.")
            print("1) You may try to use --run_locally to debug.")
            print("2) You may try to run yourself the notebook copied in "
                  "--out_dir: \n"
                  "   A) Manually change the line CONFIG_FILENAME to: \n"
                  "      {}\n"
                  "   B) Hit play."
                  .format(out_config_file))
            print("3) You may try to specify which kernel (i.e. which python "
                  "version) your jupyter notebooks should run on:\n"
                  "   >> (workon your favorite python environment)\n"
                  "   >> pip install ipykernel\n"
                  "   >> python -m ipykernel install --user\n")

        # 6. Delete config file.
        os.remove(config_filename)