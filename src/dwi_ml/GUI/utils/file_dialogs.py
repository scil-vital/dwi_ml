# -*- coding: utf-8 -*-
import logging
import os

import dearpygui.dearpygui as dpg

from dwi_ml.GUI.utils.gui_popup_message import show_infobox
from dwi_ml.GUI.styles.my_styles import INPUTS_WIDTH
from dwi_ml.GUI.styles.theme_inputs import (
    get_modified_theme, get_required_theme,
    callback_log_value_change_style_required)

home = os.path.expanduser('~')
params_file_dialogs = {'width': 700,
                       'height': 400,
                       'default_path': home,
                       'modal': True}

FILE_DIALOG_BUTTON_SUFFIX = '_button'
FILE_DIALOG_SUFFIX = '_file_dialog'


def get_file_dialog_value(arg_name):
    user_data = dpg.get_item_user_data(arg_name)
    chosen_path = user_data['chosen_path']

    return chosen_path


def _assert_single_choice_file_dialog(app_data):
    """
    Verifies that the file dialog got exactly one file.

    Return: bool, str
        bool: Wheter got exactly one file.
        str: Filename (or None)
    """
    # GUI management is very weird:
    # app_data = a dict with
    #   --> file_path_name = the current_path/file.
    #        - If nothing selected: will still be: path/.ext
    #        - If something selected: will path/file.*. You can find the
    #        correct extension in 'selections'.
    #   --> file_name: Save as above, without the path.
    #   --> current_path: Ok. Current path.
    #   --> current_filter: The filter shown. Ex: '.sh', '.*'.
    #   --> selections:
    #         - If nothing selected: Can be {}. But if you had selected
    #         something once before, and then opened it again and selected
    #         nothing, previous selection is still there.
    #         - If something selected: It is a dict with
    #               { filename.ext: path/filename.ext}
    #
    logging.debug("    File dialog: Verifying if file is a single choice: {}"
                  .format(app_data))

    # 1) Asserting no more than one
    # 'file_name' will be = '2 files Selected' if many files are selected
    # Relying on selections
    selections = list(app_data['selections'].values())
    assert len(selections) <= 1, \
        "BUG IN OUR CODE? SHOULD NOT ALLOW MULTIPLE SELECTION. " \
        "USE file_count=1, OR DO NOT USE THIS CALLBACK."

    # 2) Asserting not 0.
    if app_data['file_name'] == app_data['current_filter']:
        # There is an error in the case where the selected file starts with
        # a . (ex, .bashrc). We will suppose this will never happen. The error
        # is from dearpygui.
        show_infobox("Selection error",
                     "You did not select any file! \n\n"
                     "(If you selected a file starting with a dot, you may "
                     "also see this error. This is an issue of DearpyGUI, not "
                     "managed yet, sorry.)")
        return False, None

    # 3) Find if an existing file was chosen. selections and file_path_name are
    # the same, (except extension).
    file_path_no_ext, ext1 = os.path.splitext(app_data['file_path_name'])

    if len(selections) != 0:
        selection_no_ext, ext2 = os.path.splitext(selections[0])
    else:
        selection_no_ext = ''

    if selection_no_ext == file_path_no_ext:
        return True, selections[0]

    # 4) Else, a non-existing file was chosen (ex: where you want to save
    # some output). The file_path_name and selections are different.
    # Its extension is not there however :( If current filter is set, ok.
    # If current_filter is '.*', there is no way to know.
    # I wrote an issue here: https://github.com/hoffstadt/DearPyGui/issues/2139
    if app_data['current_filter'] == '.*':
        show_infobox("Supervision required",
                     "CAREFUL! Due to a bug in DearPyGui, we are unable to "
                     "detect the file extension when using '.*' file filter. "
                     "Correct it in the associated input text area or, in the "
                     "file dialog, select your favorite extension in the "
                     "dropdown menu on the bottom right. '.*' is not accepted "
                     "to create a new file.")
    return True, file_path_no_ext + app_data['current_filter']


def _assert_single_choice_folder_dialog(app_data):
    chosen_paths = list(app_data['selections'].values())
    assert len(chosen_paths) <= 1, \
        "BUG IN OUR CODE? SHOULD NOT ALLOW MULTIPLE SELECTION"

    if len(chosen_paths) == 0:
        show_infobox("Selection error", "You did not select any folder!")
        return False, None

    # All good
    return True, app_data['current_path']


def _finish_callback(user_data, chosen_path):
    logging.debug("Got chosen path {}. Saving and running sub-callback if "
                  "any.".format(chosen_path))
    user_data['chosen_path'] = chosen_path
    if user_data['sub_callback'] is not None:
        # Run sub-callback
        user_data['sub_callback'](chosen_path)


def _callback_single_folder_dialog(_, app_data, user_data):
    # In case the file dialog was not instantiated with file_count = 1
    ok, file_path = _assert_single_choice_folder_dialog(app_data)
    if ok:
        _finish_callback(user_data, file_path)


def _callback_single_file_dialog(_, app_data, user_data):
    # In case the file dialog was not instantiated with file_count = 1
    ok, file_path = _assert_single_choice_file_dialog(app_data)
    if ok:
        _finish_callback(user_data, file_path)


def add_single_file_dialog(tag, callback=_callback_single_file_dialog,
                           extensions=None, sub_callback=None):
    """
    If color_button: will change the associated button's color once a file is
    selected. The button must exist and be named
    tag + FILE_DIALOG_BUTTON_SUFFIX.
    """
    if extensions is None:
        extensions = []
    if '.*' not in extensions:
        extensions.append('.*')

    # Adding the 'with' makes it available instead of disappearing after this
    # method returns.
    user_data = {'chosen_path': None,
                 'sub_callback': sub_callback}
    with dpg.file_dialog(
            label="Please select a file.", callback=callback,
            show=False, directory_selector=False, tag=tag,
            file_count=1, **params_file_dialogs, user_data=user_data):
        for extension in extensions:
            dpg.add_file_extension(extension)


def add_single_folder_dialog(tag, callback=_callback_single_folder_dialog,
                             sub_callback=None):
    """
    If color_button: will change the associated button's color once a file is
    selected. The button must exist and be named
    tag + FILE_DIALOG_BUTTON_SUFFIX.
    """
    # Adding the 'with' makes it available instead of disappearing after this
    # method returns. Not sure why, but discovered it...
    user_data = {'chosen_path': None,
                 'sub_callback': sub_callback}
    with dpg.file_dialog(
            label="Please select a folder.", callback=callback,
            show=False, directory_selector=True, tag=tag,
            file_count=1, **params_file_dialogs, user_data=user_data):
        pass


# -------------------- AS A WHOLE INPUT GROUP:

def __change_input_group(sender, value):
    input_tag = sender[0:-len(FILE_DIALOG_SUFFIX)]

    # dpg.bind_item_theme(input_tag + FILE_DIALOG_BUTTON_SUFFIX,
    #                     get_modified_theme())
    dpg.set_value(input_tag, value)
    dpg.bind_item_theme(input_tag, get_modified_theme())


def __callback_single_folder_dialog_in_input_group(
        sender, app_data, user_data):
    ok, folder = _assert_single_choice_folder_dialog(app_data)
    if ok:
        __change_input_group(sender, folder)
        _finish_callback(user_data, folder)


def __callback_single_file_dialog_in_input_group(
        sender, app_data, user_data):
    ok, file_path = _assert_single_choice_file_dialog(app_data)
    if ok:
        __change_input_group(sender, file_path)
        _finish_callback(user_data, file_path)


def add_file_dialog_input_group(input_tag, file_dialog_params,
                                input_parent, button_parent):
    """
    Adds a file dialog that also shows selected file into an input text.
    Chosen selection can thus be modified by user through the input box.

    Parameters
    ----------
    input_tag: str
    file_dialog_params: list: [str, list]
        - type: Either 'unique_file' or 'unique_folder'. Eventually, multi-
                selection could be accepted. Not implemented yet.
        - ext: The list of accepted extensions.
    input_parent: str
        Where to place the input text.
    button_parent: str
        Where to place the click button.
    """
    file_dialog_type = file_dialog_params[0]

    # 1. Create hidden file dialog.
    if file_dialog_type == 'unique_file':
        file_dialog_ext = file_dialog_params[1]
        callback = __callback_single_file_dialog_in_input_group
        add_single_file_dialog(input_tag + FILE_DIALOG_SUFFIX,
                               callback=callback, extensions=file_dialog_ext)
    elif file_dialog_type == 'unique_folder':
        callback = __callback_single_folder_dialog_in_input_group
        add_single_folder_dialog(input_tag + FILE_DIALOG_SUFFIX,
                                 callback=callback)
    else:
        raise NotImplementedError("Unkown file dialog type.")

    # 2. Add input + button
    item = dpg.add_input_text(
        tag=input_tag, width=INPUTS_WIDTH - 100,
        callback=callback_log_value_change_style_required, parent=input_parent)
    dpg.add_button(
        tag=input_tag + FILE_DIALOG_BUTTON_SUFFIX, width=100,
        label='Click to select.',
        callback=lambda: dpg.show_item(input_tag + FILE_DIALOG_SUFFIX),
        parent=button_parent)

    dpg.bind_item_theme(item, get_required_theme())
