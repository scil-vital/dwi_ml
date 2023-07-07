# -*- coding: utf-8 -*-
import os

import dearpygui.dearpygui as dpg

home = os.path.expanduser('~')
params_file_dialogs = {'width': 700,
                       'height': 400,
                       'file_count': 1,
                       'default_path': home,
                       'modal': True
                       }


def open_file_dialog_hdf5_file():
    if dpg.does_item_exist("file_dialog_hdf5_file"):
        dpg.show_item("file_dialog_hdf5_file")
    else:
        dpg.add_file_dialog(
            directory_selector=False, tag="file_dialog_hdf5_file",
            **params_file_dialogs, callback=UNDEFINED_YET)


def open_file_dialog_experiments_path():
    if dpg.does_item_exist("file_dialog_experiments_path"):
        dpg.show_item("file_dialog_experiments_path")
    else:
        dpg.add_file_dialog(
            directory_selector=True, tag="file_dialog_experiments_path",
            **params_file_dialogs, callback=UNDEFINED_YET)
