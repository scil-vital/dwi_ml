# -*- coding: utf-8 -*-
import os

from dwi_ml.gui.utils.gui_popup_message import show_infobox

home = os.path.expanduser('~')
params_file_dialogs = {'width': 700,
                       'height': 400,
                       'default_path': home,
                       'modal': True}


def _assert_single_choice_file_dialog(app_data):
    chosen_paths = list(app_data['selections'].values())
    if len(chosen_paths) > 1:
        raise NotImplementedError("BUG IN OUR CODE? SHOULD NOT ALLOW "
                                  "MULTIPLE SELECTION")
    if len(chosen_paths) == 0:
        return False
    return True


def callback_file_dialog_single_file(_, app_data, sub_callback):
    """
    user_data: optional, Callable
        sub-callback method. Will be called with chosen path.
    """
    # In case the file dialog was not instantiated with file_count = 1
    ok = _assert_single_choice_file_dialog(app_data)
    if ok:
        chosen_path = app_data['current_path']
        if sub_callback is not None:
            sub_callback(chosen_path)
    else:
        show_infobox("Selection error", "You did not select a directory!")
