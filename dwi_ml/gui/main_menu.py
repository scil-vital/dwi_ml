# -*- coding: utf-8 -*-
import os
import time

import dearpygui.dearpygui as dpg

from dwi_ml.gui.l2t_menus import open_l2t_from_checkpoint_window, \
    prepare_and_show_train_l2t_window
from dwi_ml.gui.transformers_menus import open_tto_from_checkpoint_window, \
    open_ttst_from_checkpoint_window, open_tto_window, open_ttst_window
from dwi_ml.gui.utils.gui_popup_message import show_infobox, \
    simply_close_infobox
from dwi_ml.gui.utils.inputs import assert_single_choice_file_dialog
from dwi_ml.gui.utils.my_styles import get_my_fonts_dictionary
from dwi_ml.gui.utils.window import start_dpg, show_and_end_dpg
from dwi_ml.io_utils import verify_checkpoint_exists, verify_which_model_in_path


def file_dialog_ok_callback(sender, app_data):
    assert_single_choice_file_dialog(app_data)
    chosen_path = app_data['current_path']
    if sender == "file_dialog_resume_from_checkpoint":
        open_checkpoint_subwindow(chosen_path)


def file_dialog_cancel_callback(_, __):
    pass


def prepare_main_menu():
    start_dpg()

    # ---------
    # Preparing file dialgogs. There seems to be a bug with dearpygui, I cannot
    # separate this from main code (i.e. encapsulate).
    # See my issue. https://github.com/hoffstadt/DearPyGui/issues/2120
    # ---------
    home = os.path.expanduser('~')

    # Directory dialogs
    params = {'width': 700,
              'height': 400,
              'file_count': 1,
              'callback': file_dialog_ok_callback,
              'cancel_callback': file_dialog_cancel_callback,
              'default_path': home,
              'show': False,
              'modal': True
              }
    dpg.add_file_dialog(
        directory_selector=True, tag="file_dialog_experiments_path", **params)
    dpg.add_file_dialog(
        directory_selector=True, tag="file_dialog_resume_from_checkpoint",
        **params)
    dpg.add_file_dialog(
        directory_selector=False, tag="file_dialog_hdf5_file", **params)
    dpg.add_file_dialog(
        directory_selector=False, tag="file_dialog_output_script", **params)

    # ---------
    # Ok. Now preparing the window
    # ---------
    titles = []
    with dpg.window(tag="Primary Window"):
        main_title = dpg.add_text("                                           "
                                  "                     WELCOME TO DWI_ML")

        ##########
        # 1. Preparing the HDF5
        ##########
        titles.append(dpg.add_text("\n\nPrepare your data:"))
        dpg.add_button(label="Create your hdf5 (to come)", indent=50)

        ##########
        # 2. Training
        ##########
        titles.append(dpg.add_text("\n\nTrain a new model:"))
        dpg.add_button(label="Learn2track", indent=50,
                       callback=prepare_and_show_train_l2t_window)
        dpg.add_button(label="Transforming tractography: original model",
                       indent=50, callback=open_tto_window)
        dpg.add_button(label="Transforming tractography: source-target model",
                       indent=50, callback=open_ttst_window)

        ##########
        # 3. Resume from checkpoint.
        ##########
        titles.append(dpg.add_text(
            "\n\nContinue training model from an existing experiment "
            "(resume from checkpoint):"))
        dpg.add_button(label="Select your experiment's directory",
                       callback=lambda: dpg.show_item("file_dialog_resume_from_checkpoint"),
                       indent=50)

        ##########
        # 4. Visualize logs
        ##########
        titles.append(dpg.add_text("\n\nVisualization"))
        dpg.add_button(label="(To come)", indent=50)

        ##########
        # 5. Track from a model.
        ##########
        titles.append(dpg.add_text("\n\nTrack from a model"))
        dpg.add_button(label="(To come)", indent=50)

        my_fonts = get_my_fonts_dictionary()
        dpg.bind_font(my_fonts['default'])
        dpg.bind_item_font(main_title, my_fonts['main_title'])
        for title in titles:
            dpg.bind_item_font(title, my_fonts['title'])

    show_and_end_dpg()


def open_checkpoint_subwindow(chosen_path):
    # toDo: if raises a FileNotFoundError, show a pop-up warning?
    #  Currently, prints the warning in terminal.
    try:
        checkpoint_path = verify_checkpoint_exists(chosen_path)
    except FileNotFoundError:
        print("SHOULD SHOW A WARNING")
        show_infobox("Wrong experiment path!",
                     "No checkpoint folder found in this directory! Is this "
                     "really an experiment path? Please select another one."
                     "\n\n"
                     "(Chosen path: {})".format(chosen_path))
        return

    model_dir = os.path.join(checkpoint_path, 'model')
    model_type = verify_which_model_in_path(model_dir)
    print("Model's class: {}".format(model_type))

    if model_type == 'Learn2TrackModel':
        open_l2t_from_checkpoint_window()
    elif model_type == 'OriginalTransformerModel':
        open_tto_from_checkpoint_window()
    elif model_type == 'TransformerSrcAndTgtModel':
        open_ttst_from_checkpoint_window()
    else:
        raise ValueError("This type of model is not managed by our DWIML' GUI.")
