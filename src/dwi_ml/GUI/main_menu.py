# -*- coding: utf-8 -*-
import logging

import dearpygui.dearpygui as dpg

from dwi_ml.GUI.menu1_create_hdf5 import open_menu_create_hdf5
from dwi_ml.GUI.menu1_l2t import open_menu_learn2track
from dwi_ml.GUI.menu1_tt import open_menu_transformers
from dwi_ml.GUI.styles.my_styles import get_my_fonts_dictionary


def prepare_main_menu():
    logging.debug("GUI.main_menu.prepare_main_menu(): "
                  "Preparing main window...")

    titles = []
    with dpg.window(tag="Primary Window"):
        main_title = dpg.add_text(
            "                                                                "
            "WELCOME TO DWI_ML")
        dpg.add_text("\nSelect a sub-menu below based on your needs. \n"
                     "All menus have as objective to help you create a bash "
                     "script (my_file.sh) with all your favorite options.")

        # 1. Button to get to menu "Preparing the HDF5"
        titles.append(dpg.add_text("\n\nPrepare your data as a hdf5 file:"))
        dpg.add_button(label="Create your hdf5", indent=50,
                       callback=open_menu_create_hdf5)

        # 2. Button to get to menu "Learn2track"
        titles.append(dpg.add_text("\n\nGet to Learn2track's menu:"))
        dpg.add_button(label="Learn2track", indent=50,
                       callback=open_menu_learn2track)

        # 3. Button to get to menu "Transformers"
        titles.append(dpg.add_text("\n\nGet to Transforming Tractography's "
                                   "menu:"))
        dpg.add_button(label="Transforming tractography",
                       indent=50, callback=open_menu_transformers)

    # Set Title fonts
    my_fonts = get_my_fonts_dictionary()
    dpg.bind_font(my_fonts['default'])
    dpg.bind_item_font(main_title, my_fonts['main_title'])
    for title in titles:
        dpg.bind_item_font(title, my_fonts['section_title'])

    dpg.set_primary_window("Primary Window", True)
