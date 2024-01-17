# -*- coding: utf-8 -*-
import os
import shutil
import subprocess
import sys
from os.path import dirname

from scilpy.io.fetcher import get_home as get_scilpy_folder
from scilpy.io.utils import assert_outputs_exist

from dwi_ml.testing.projects.tt_visu_main import (
    build_argparser_transformer_visu, create_out_dir_visu_weights,
    tt_visualize_weights_main)


# Note. To use through jupyter, the file must also be up-to-date.
# To modify it as developers:
# jupyter notebook dwi_ml/testing/projects/tt_visualise_weights.ipynb
# Do not run it there though! Else, saving it will save the outputs too!


def _get_config_filename():
    """
    File that will be saved by the python script with all the args. The
    jupyter notebook can then load them again.
    """
    # We choose to add it in the hidden .scilpy folder in our home.
    # (Where our test data also is).
    hidden_folder = get_scilpy_folder()
    config_filename = os.path.join(
        hidden_folder, 'ipynb_tt_visualize_weights.config')
    return config_filename


def main():
    # Getting the raw args. Will be sent to jupyter.
    argv = sys.argv

    parser = build_argparser_transformer_visu()
    args = parser.parse_args()

    # Verifying if jupyter is required.
    if 'bertviz' in args.visu_type and 'bertviz_locally' in args.visu_type:
        raise ValueError("Please only select 'bertviz' or 'bertviz_locally', "
                         "not both.")
    elif 'bertviz_locally' in args.visu_type:
        print("--DEBUGGING MODE--\n"
              "We will run the bertviz but it will not save any output!")
        run_locally = True
    elif 'bertviz' in args.visu_type:
        print("Preparing to run through jupyter.")
        run_locally = False
    else:
        run_locally = True

    # Running.
    if run_locally:
        tt_visualize_weights_main(args, parser)
    else:
        # PREPARING TO RUN THROUGH JUPYTER

        # 1) Finding the jupyter notebook
        dwi_ml_dir = dirname(dirname(__file__))
        raw_ipynb_filename = os.path.join(
            dwi_ml_dir, 'dwi_ml/testing/projects/tt_visualize_weights.ipynb')
        if not os.path.isfile(raw_ipynb_filename):
            raise ValueError(
                "We could not find the jupyter notebook file. Probably a "
                "coding error on our side. We expected it to be in {}"
                .format(raw_ipynb_filename))

        # 2) Verify that output dir exists but not the html output files.
        args = create_out_dir_visu_weights(args)

        out_html_filename = args.out_prefix + 'tt_bertviz.html'
        out_html_file = os.path.join(args.out_dir, out_html_filename)
        out_ipynb_file = os.path.join(
            args.out_dir, args.out_prefix + 'tt_bertviz.ipynb')
        out_config_file = os.path.join(
            args.out_dir, args.out_prefix + 'tt_bertviz.config')
        assert_outputs_exist(parser, args,
                             [out_html_file, out_ipynb_file, out_config_file])

        # 2) Save the args to this script in a config file, to be read again
        # by the notebook in a new argparse instance.
        # Jupyter notebook needs to know where to load the config file.
        # Needs to be always at the same place because we cannot send an
        # argument to jupyter.
        hidden_config_filename = _get_config_filename()
        if os.path.isfile(hidden_config_filename):
            # In case a previous call failed.
            os.remove(hidden_config_filename)
        with open(hidden_config_filename, 'w') as f:
            f.write(' '.join(argv))

        # 4. Copy files locally so that user can see it.
        shutil.copyfile(raw_ipynb_filename, out_ipynb_file)
        shutil.copyfile(hidden_config_filename, out_config_file)

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
                .format(args.out_dir, out_html_filename, raw_ipynb_filename)
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
        os.remove(hidden_config_filename)
