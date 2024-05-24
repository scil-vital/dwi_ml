# -*- coding: utf-8 -*-

import dearpygui.dearpygui as dpg

from dwi_ml.GUI.styles.my_styles import (
    WINDOW_WIDTH, _create_item_theme_with_bckground_color)

"""
Sets the colors for the input boxes / text / etc.
- When required and not yet modified: red
- When non required and not yet modified: gray
- When already set: turns purple.
"""
chosen_purple = (130, 75, 177, 255)
required_red = (242, 8, 8, 255)
gray = (78, 78, 78, 255)


# Defining values that will be used as global constants to avoid re-defining
# many times the same theme for each item.
global_modified_theme = None
global_non_modified_theme = None
global_required_theme = None


def add_color_legend():
    dpg.add_text("Values in red are required",
                 color=required_red, indent=WINDOW_WIDTH - 300)
    dpg.add_text("Values in purple already have been set.",
                 color=chosen_purple, indent=WINDOW_WIDTH - 300)


def get_modified_theme() -> int:
    global global_modified_theme
    if global_modified_theme is None:
        global_modified_theme = _create_item_theme_with_bckground_color(chosen_purple)
    return global_modified_theme


def get_non_modified_theme() -> int:
    global global_non_modified_theme
    if global_non_modified_theme is None:
        global_non_modified_theme = _create_item_theme_with_bckground_color(gray)
    return global_non_modified_theme


def get_required_theme() -> int:
    global global_required_theme
    if global_required_theme is None:
        global_required_theme = _create_item_theme_with_bckground_color(required_red)
    return global_required_theme


def callback_log_value_change_style_required(sender, app_data, _):
    if str(dpg.get_item_type(sender)) == 'mvAppItemType::mvCheckbox':
        # Checkbox.
        raise NotImplementedError("We have not prepared this callback for "
                                  "checkbox values.")
    else:
        if app_data is None or app_data == '':
            # User has deleted the value. Required again.
            dpg.bind_item_theme(sender, get_required_theme())
        else:
            # User has entered some value
            dpg.bind_item_theme(sender, get_modified_theme())

