#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dwi_ml.gui.menu_new_or_exisiting import launch_main_menu
from dwi_ml.gui.utils.window import start_dpg, show_and_end_dpg


def main():

    start_dpg()
    launch_main_menu()
    show_and_end_dpg()


if __name__ == '__main__':
    main()


