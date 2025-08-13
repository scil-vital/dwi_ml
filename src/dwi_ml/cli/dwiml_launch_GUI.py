#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Launch the GUI to help you prepare a script with all your favorite options.
"""
import argparse
import logging

from scilpy.io.utils import add_verbose_arg

from dwi_ml.GUI.main_menu import prepare_main_menu
from dwi_ml.GUI.utils.window import start_dpg, show_and_end_dpg

# import dearpygui.demo as demo
# demo.show_demo()
# import dearpygui.dearpygui as dpg
# dpg.show_style_editor()


def _prepare_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    add_verbose_arg(p)
    return p


def main():
    p = _prepare_argparser()
    args = p.parse_args()
    logging.getLogger().setLevel(args.verbose)

    start_dpg()
    prepare_main_menu()

    show_and_end_dpg()


if __name__ == '__main__':
    main()


