# -*- coding: utf-8 -*-
from typing import Optional

import dearpygui.dearpygui as dpg

from dwi_ml.gui.utils.my_styles import STYLE_INPUT_ITEM, get_required_theme, \
    get_none_theme, get_non_modified_theme, STYLE_INPUT_CHECKBOX_ITEM, \
    get_modified_theme

CHECKBOX_SUFFIX = '_default_checkbox'


def change_none_to_gui_default(value, dtype):
    """
    Usage: when a default value for a GUI input would be None (which is not
    possible for the GUI), change its value to a 'normal' default value, based
    on the data type.
    """
    if value is not None:
        # Assert that default is the right type?
        return value
    else:
        if dtype == str:
            return ''
        elif dtype == int:
            return 0
        elif dtype == float:
            return 0.0
        else:
            raise ValueError("Data type {}' default not supported yet!"
                             .format(dtype))


def _log_value_verify_exclusive_group_and_checkbox(sender, app_data, user_data):
    """
    Callback intended for input items belonging to a mutually exclusive group.
    By default, they are always optional and thus are associated to a default
    checkbox.
    """
    elements_in_group, has_check = user_data

    # Changing background color
    if str(dpg.get_item_type(sender)) == 'mvAppItemType::mvCheckbox':
        # Checkbox. If unset: unmodified
        if app_data:
            dpg.bind_item_theme(sender, get_modified_theme())
        else:
            dpg.bind_item_theme(sender, get_non_modified_theme())
    else:
        dpg.bind_item_theme(sender, get_modified_theme())

    # Removing check.
    if has_check:
        dpg.set_value(sender + CHECKBOX_SUFFIX, False)

    # Re-checking (to None) other elements.
    if elements_in_group is not None:
        for a in elements_in_group:
            if a != sender:
                # Other elements are always optional.
                if dpg.does_item_exist(a + CHECKBOX_SUFFIX):
                    # But either they have a checkbox:
                    dpg.set_value(a + CHECKBOX_SUFFIX, True)
                else:
                    # Or they are themselves a checkbox (ex, 'store_true')
                    # Unsetting.
                    dpg.set_value(a, False)
                dpg.bind_item_theme(a, get_non_modified_theme())


def add_input_item_based_on_type(arg_name: str, params, required: bool,
                                 exclusive_group: Optional[list],
                                 has_check: bool):
    """
    Add an input item based on datatype:
    - A dropdown list of choices
    - Boolean value: currently a dropdown with True/False. Could be a checkbox,
        was not as nice visually.
    - Int/float value: A box with value and arrows on the side to
        increase/decrease chosen input.
    - Str: A simple input box.

    Parameters
    ----------
    arg_name: str
        Will be the tag of the input
    params: dict.
        We will verify if it contains key 'choices', and key 'action'
        (store_true, store_false), and key 'dtype' (else the data type is str).
    required: bool
        Modifying item theme based on if it is required.
    exclusive_group: list
        List of other inter-related args.
    has_check: bool
    """
    default = None if 'default' not in params else params['default']
    dtype = params['type'] if 'type' in params else str

    item_options = {
        'tag': arg_name,
        'callback': _log_value_verify_exclusive_group_and_checkbox,
        'user_data': (exclusive_group, has_check)
    }

    if 'choices' in params:
        choices = list(params['choices'])
        gui_default = choices[0] if default is None else default
        item = dpg.add_combo(choices, default_value=gui_default,
                             **STYLE_INPUT_ITEM, **item_options)

    elif 'action' in params:
        if params['action'] in ['store_true', 'store_false']:
            default = False
            item = dpg.add_checkbox(
                default_value=default,
                **STYLE_INPUT_CHECKBOX_ITEM, **item_options)
        else:
            raise NotImplementedError("ACTION {} NOT MANAGED YET"
                                      .format(params['action']))
    elif dtype == bool:
        tmp_default = str(default) if default is not None else None
        item = dpg.add_combo(['True', 'False'], default_value=tmp_default,
                             **STYLE_INPUT_ITEM, **item_options)

    elif dtype == str:
        item = dpg.add_input_text(
            default_value=change_none_to_gui_default(default, str),
            **STYLE_INPUT_ITEM, **item_options)

    elif dtype == int:
        item = dpg.add_input_int(
            default_value=change_none_to_gui_default(default, int),
            **STYLE_INPUT_ITEM, **item_options)

    elif dtype == float:
        item = dpg.add_input_float(
            default_value=change_none_to_gui_default(default, float),
            format='%.7f', **STYLE_INPUT_ITEM, **item_options)

    else:
        item = dpg.add_text("NOT MANAGED YET TYPE {} FOR ARG {}: "
                            .format(dtype, arg_name),
                            **STYLE_INPUT_ITEM, **item_options)

    if default is None:
        if required:
            dpg.bind_item_theme(item, get_required_theme())
        else:
            dpg.bind_item_theme(item, get_none_theme())
    else:
        dpg.bind_item_theme(item, get_non_modified_theme())

    return dtype, default
