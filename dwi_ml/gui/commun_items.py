# -*- coding: utf-8 -*-

import dearpygui.dearpygui as dpg

from dwi_ml.gui.utils.file_dialogs import params_file_dialogs, \
    callback_file_dialog_single_file


# ----------- Common experiment args:
def open_file_dialog_hdf5_file(sub_callback=None):
    """
    sub_callback: Callable
        Method. Will be called with chosen path from new file dialog.
    """
    file_dialog_name = "file_dialog_hdf5_file"
    if dpg.does_item_exist(file_dialog_name):
        dpg.show_item(file_dialog_name)
    else:
        with dpg.file_dialog(
                label="Please select your HDF5 file",
                directory_selector=False, tag=file_dialog_name,
                file_count=1, **params_file_dialogs,
                callback=callback_file_dialog_single_file,
                user_data=sub_callback):
            dpg.add_file_extension(".hdf5", custom_text="[hdf5 file]")


def open_file_dialog_experiments_path(sub_callback=None):
    file_dialog_name = "file_dialog_experiments_path"
    if dpg.does_item_exist(file_dialog_name):
        dpg.show_item(file_dialog_name)
    else:
        dpg.add_file_dialog(
            directory_selector=True, tag=file_dialog_name,
            file_count=1, **params_file_dialogs,
            callback=callback_file_dialog_single_file,
            user_data=sub_callback)


# ----------- Common GUI buttons:
def open_file_dialog_script_path():


def add_button_get_arg_values_save_bash_script(args, save_script_method):
    with dpg.group(horizontal=True):
        dpg.add_button(label="Choose bash script path.")
        dpg.add_text(hint="my_file.sh")
        dpg.add_button(label='Create my script!', indent=1000,
                       callback=callback_ok_get_args_l2t,
                       user_data=args, height=50)
