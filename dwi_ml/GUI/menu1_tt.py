# -*- coding: utf-8 -*-
import logging

import dearpygui.dearpygui as dpg

from dwi_ml.GUI.styles.my_styles import (STYLE_FIXED_WINDOW,
                                         get_my_fonts_dictionary)
from dwi_ml.GUI.utils.window import callback_change_window

TAG_HDF_SCRIPT_PATH = 'hdf5_creation_script'


def open_menu_transformers():
    logging.debug("GUI.menu1_tt.open_menu_transformers(): "
                  "Preparing TT window...")
    main_window = dpg.get_active_window()
    dpg.hide_item(main_window)

    titles = []
    if dpg.does_item_exist('tt_window'):
        # User had already reached this page but then left and came back.
        dpg.set_primary_window('tt_window', True)
        dpg.show_item('tt_window')
    else:
        # Create the L2T window.
        with dpg.window(**STYLE_FIXED_WINDOW,
                        tag='tt_window') as tt_window:
            dpg.set_primary_window(tt_window, True)

            dpg.add_button(label='<-- Back', callback=callback_change_window,
                           user_data=(tt_window, main_window))

            main_title = dpg.add_text(
                "                                                            "
                "    TRANSFORMING TRACTOGRAPHY:\n\n")
            titles.append(dpg.add_text("Choose your next step."))

            # 1. Button to get to menu "Training"
            dpg.add_button(label="Train a new model", indent=50,
                           callback=open_train_tt_window)

            # 2. Button to get to menu "Continue from checkpoint"
            dpg.add_button(label="Continue from checkpoint", indent=50,
                           callback=open_train_tt_window)

            # 3. Button to get to menu "Track from model"
            dpg.add_button(label="Track from model", indent=50,
                           callback=open_train_tt_window)

            # 4. Button to get to menu "Visualize loss"
            dpg.add_button(label="Visualize loss", indent=50,
                           callback=open_train_tt_window)

            # 5. Button to get to menu "Visualize weights evolution"
            # toDo Script not added to tests!! Could be heavily deprecated.
            dpg.add_button(label="Visualize weights evolution", indent=50,
                           callback=open_train_tt_window)

        # Set Title fonts
        my_fonts = get_my_fonts_dictionary()
        dpg.bind_font(my_fonts['default'])
        dpg.bind_item_font(main_title, my_fonts['main_title'])
        for title in titles:
            dpg.bind_item_font(title, my_fonts['section_title'])

    logging.debug("...TT window done.\n")
