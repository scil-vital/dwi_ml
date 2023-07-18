# -*- coding: utf-8 -*-
import os

import dearpygui.dearpygui as dpg

from dwi_ml.gui.utils.gui_popup_message import show_infobox
from dwi_ml.gui.utils.my_styles import get_modified_theme

home = os.path.expanduser('~')
params_file_dialogs = {'width': 700,
                       'height': 400,
                       'default_path': home,
                       'modal': True}


def _assert_single_choice_file_dialog(app_data):
    """
    Verifies that the file dialog got exactly one file.
    Return: bool, str
        bool: Wheter got exactly one file.
        str: Filename (or None)
    """
    # 1) Asserting no more than one
    # 'file_name' will be = '2 files Selected' if many files are selected
    selections = list(app_data['selections'].values())
    if len(selections) > 1:
        raise NotImplementedError("BUG IN OUR CODE? SHOULD NOT ALLOW "
                                  "MULTIPLE SELECTION. USE file_count=1, OR "
                                  "DO NOT USE THIS CALLBACK.")

    # 2) Asserting not 0.
    # If no file was chosen, 'file_name', will be empty, or will be only,
    # for instance, '.hdf5'.
    if len(selections) == 0:
        return False, None

    # All good.
    return True, selections[0]


def _assert_single_choice_folder_dialog(app_data):
    chosen_paths = list(app_data['selections'].values())
    if len(chosen_paths) > 1:
        raise NotImplementedError("BUG IN OUR CODE? SHOULD NOT ALLOW "
                                  "MULTIPLE SELECTION")
    if len(chosen_paths) == 0:
        return False, None

    # All good
    return True, app_data['current_path']


def callback_single_file_dialog(sender, app_data, results):
    """
    user_data: A dict with a single value:
        { tag: result }
    """
    # In case the file dialog was not instantiated with file_count = 1
    ok, file_path = _assert_single_choice_file_dialog(app_data)
    if not ok:
        show_infobox("Selection error", "You did not select any file!")
    else:
        dpg.bind_item_theme(sender + '_button', get_modified_theme())
    results[sender] = file_path


def callback_single_folder_dialog(sender, app_data, results):
    """
    user_data: A dict with a single value:
        { tag: result }
    """
    # In case the file dialog was not instantiated with file_count = 1
    ok, folder = _assert_single_choice_folder_dialog(app_data)
    if not ok:
        show_infobox("Selection error", "You did not select any folder!")
    else:
        dpg.bind_item_theme(sender + '_button', get_modified_theme())
    results[sender] = folder


def add_single_file_dialog(tag, extensions=None):
    if extensions is None:
        extensions = []
    if '.*' not in extensions:
        extensions.append('.*')

    # Adding the 'with' makes it available instead of disappearing after this
    # method returns.
    results = {tag: None}
    with dpg.file_dialog(
            label="Please select a file.",
            show=False, directory_selector=False, tag=tag,
            file_count=1, **params_file_dialogs, user_data=results,
            callback=callback_single_file_dialog):
        for extension in extensions:
            dpg.add_file_extension(extension)


def add_single_folder_dialog(tag):
    # Adding the 'with' makes it available instead of disappearing after this
    # method returns.
    results = {tag: None}
    with dpg.file_dialog(
            label="Please select a folder.",
            show=False, directory_selector=True, tag=tag,
            file_count=1, **params_file_dialogs, user_data=results,
            callback=callback_single_folder_dialog):
        pass
