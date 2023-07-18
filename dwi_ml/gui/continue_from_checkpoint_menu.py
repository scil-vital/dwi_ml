# -*- coding: utf-8 -*-
import os

import dearpygui.dearpygui as dpg

from dwi_ml.gui.projects.l2t_menus import open_l2t_from_checkpoint_window
from dwi_ml.gui.projects.l2t_from_checkpoint_menu import \
    open_l2t_from_checkpoint_window
from dwi_ml.gui.projects.transformers_menus import open_tto_from_checkpoint_window, \
    open_ttst_from_checkpoint_window
from dwi_ml.gui.utils.file_dialogs import params_file_dialogs, \
    callback_file_dialog_single_file
from dwi_ml.gui.utils.gui_popup_message import show_infobox
from dwi_ml.io_utils import verify_checkpoint_exists, verify_which_model_in_path


def show_file_dialog_from_checkpoint():
    file_dialog_name = "file_dialog_resume_from_checkpoint"
    if dpg.does_item_exist(file_dialog_name):
        dpg.show_item(file_dialog_name)
    else:
        dpg.add_file_dialog(
            directory_selector=True, tag=file_dialog_name,
            file_count=1, **params_file_dialogs,
            callback=callback_file_dialog_single_file,
            user_data=_open_checkpoint_subwindow)


def _open_checkpoint_subwindow(chosen_path):
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
