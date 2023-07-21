#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

from dwi_ml.gui.main_menu import prepare_main_menu
from dwi_ml.gui.utils.window import start_dpg, show_and_end_dpg


def main():
    logging.getLogger().setLevel(level='WARNING')
    start_dpg()
    prepare_main_menu()
    show_and_end_dpg()


if __name__ == '__main__':
    main()


