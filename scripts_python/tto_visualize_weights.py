# -*- coding: utf-8 -*-
import shutil
import sys
import os
from os.path import join, dirname, isfile, isdir

import subprocess

from dwi_ml.models.projects.transformer_visualisation_utils import \
    build_argparser_transformer_visu, tto_visualize_weights

dwi_ml_dir = dirname(dirname(__file__))
config_dir = join(dwi_ml_dir, '.ipynb_config')
if not isdir(config_dir):
    os.mkdir(config_dir)
IPYNB_FILENAME = join(
    dwi_ml_dir, 'dwi_ml/models/projects/tto_visualize_weights.ipynb')
CONFIG_FILENAME = join(
    config_dir, '.config_ipynb_tto_visualize_weights')
HTML_FILENAME = 'tto_visualize_weights.html'


def main(argv):

    parser = build_argparser_transformer_visu()
    args = parser.parse_args()

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

        # 2. Verify that output dir exists.
        if not isdir(args.out_html_dir):
            raise NotADirectoryError("HTML output directory {} not found."
                                     .format(args.out_html_dir))
        if isfile(join(args.out_html_dir, HTML_FILENAME)) and \
                '-f' not in argv:
            raise FileExistsError("File {} already exists in directory {}. "
                                  "Please delete"
                                  .format(HTML_FILENAME, args.out_html_dir))

        # 3. Save the args to this script in a config file, to be read again
        # by the notebook in a new argparser instance.
        print("Writing config_file in {}")
        if isfile(CONFIG_FILENAME):
            os.remove(CONFIG_FILENAME)
        with open(CONFIG_FILENAME, 'w') as f:
            f.write(' '.join(argv))

        # 4. Launching.
        print("\n\n"
              "** We will by running the jupyter notebook and save its result "
              "as a html file in {} **\n\n".format(args.out_html_dir))
        try:
            command = 'jupyter nbconvert --output-dir={:s} --execute {:s} ' \
                      '--to html'.format(args.out_html_dir, IPYNB_FILENAME)
            print("Running command:\n\n>>{}\n\n".format(command))
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError:
            print("JUPYTER NOTEBOOK EXCEUTION WAS NOT SUCCESSFULL.")
            print("You may try to use --run_locally to debug.")
            print("You may try to run yourself the notebook copied in "
                  "--out_html_dir using the config file "
                  ".config_ipynb_tto_visualize_weights also copied there.")

        # 5. Copy locally and delete config file.
        print("Saving raw files in {}".format(args.out_html_dir))
        shutil.copyfile(
            CONFIG_FILENAME,
            join(args.out_html_dir, '.config_ipynb_tto_visualize_weights'))
        shutil.copyfile(
            IPYNB_FILENAME,
            join(args.out_html_dir, 'tto_visualize_weights.ipynb'))
        os.remove(CONFIG_FILENAME)


if __name__ == '__main__':
    main(sys.argv)

