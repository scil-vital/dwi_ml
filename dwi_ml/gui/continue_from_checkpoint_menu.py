# -*- coding: utf-8 -*-
import os

import dearpygui.dearpygui as dpg

from dwi_ml.gui.l2t_menus import open_l2t_from_checkpoint_window
from dwi_ml.gui.transformers_menus import open_tto_from_checkpoint_window, \
    open_ttst_from_checkpoint_window
from dwi_ml.gui.utils.file_dialogs import params_file_dialogs
from dwi_ml.gui.utils.gui_popup_message import show_infobox
from dwi_ml.io_utils import verify_checkpoint_exists, verify_which_model_in_path


def file_dialog_callback_checkpoint(sender, app_data):
    # assert_single_choice_file_dialog(app_data)
    chosen_path = app_data['current_path']
    _open_checkpoint_subwindow(chosen_path)


def show_file_dialog_from_checkpoint():
    if dpg.does_item_exist("file_dialog_resume_from_checkpoint"):
        dpg.show_item("file_dialog_resume_from_checkpoint")
    else:
        dpg.add_file_dialog(
            directory_selector=True, tag="file_dialog_resume_from_checkpoint",
            **params_file_dialogs, callback=file_dialog_callback_checkpoint)


def _open_checkpoint_subwindow(chosen_path):
    # toDo: if raises a FileNotFoundError, show a pop-up warning?
    #  Currently, prints the warning in terminal.
    try:
        checkpoint_path = verify_checkpoint_exists(chosen_path)
    except FileNotFoundError:
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
