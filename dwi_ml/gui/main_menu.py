# -*- coding: utf-8 -*-


import dearpygui.dearpygui as dpg

from dwi_ml.gui.continue_from_checkpoint_menu import \
    show_file_dialog_from_checkpoint
from dwi_ml.gui.create_hdf5_menu import open_menu_create_hdf5
from dwi_ml.gui.l2t_menus import open_train_l2t_window
from dwi_ml.gui.transformers_menus import open_tto_window, open_ttst_window
from dwi_ml.gui.utils.my_styles import get_my_fonts_dictionary


def prepare_main_menu():

    titles = []
    with dpg.window(tag="Primary Window"):
        main_title = dpg.add_text(
            "                                                                "
            "WELCOME TO DWI_ML")

        # 1. Preparing the HDF5
        titles.append(dpg.add_text("\n\nPrepare your data:"))
        dpg.add_button(label="Create your hdf5", indent=50,
                       callback=open_menu_create_hdf5)

        # 2. Training
        titles.append(dpg.add_text("\n\nTrain a new model:"))
        dpg.add_button(label="Learn2track", indent=50,
                       callback=open_train_l2t_window)
        dpg.add_button(label="Transforming tractography: original model",
                       indent=50, callback=open_tto_window)
        dpg.add_button(label="Transforming tractography: source-target model",
                       indent=50, callback=open_ttst_window)

        # 3. Resume from checkpoint.
        titles.append(dpg.add_text(
            "\n\nContinue training model from an existing experiment "
            "(resume from checkpoint):"))
        dpg.add_button(label="Select your experiment's directory",
                       callback=show_file_dialog_from_checkpoint, indent=50)

        # 4. Visualize logs
        titles.append(dpg.add_text("\n\nVisualization"))
        dpg.add_button(label="(To come)", indent=50)

        # 5. Track from a model.
        titles.append(dpg.add_text("\n\nTrack from a model"))
        dpg.add_button(label="(To come)", indent=50)

    # Set Title fonts
    my_fonts = get_my_fonts_dictionary()
    dpg.bind_font(my_fonts['default'])
    dpg.bind_item_font(main_title, my_fonts['main_title'])
    for title in titles:
        dpg.bind_item_font(title, my_fonts['title'])

    dpg.set_primary_window("Primary Window", True)
