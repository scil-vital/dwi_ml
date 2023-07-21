# -*- coding: utf-8 -*-

import dearpygui.dearpygui as dpg

from dwi_ml.gui.utils.my_styles import STYLE_INPUT_ITEM, get_required_theme, \
    get_none_theme, get_non_modified_theme


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


def add_input_item_based_on_type(arg_name, params, required,
                                 callback_options=None):
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
    - arg_name: will be the tag of the input
    - params: dict.
        We will verify if it contains key 'choices', and key 'action'
        (store_true, store_false), and key 'dtype' (else the data type is str).
    """
    if callback_options is None:
        callback_options = {}

    default = None if 'default' not in params else params['default']
    dtype = params['type'] if 'type' in params else str

    if 'choices' in params:
        choices = list(params['choices'])
        gui_default = choices[0] if default is None else default
        item = dpg.add_combo(choices, tag=arg_name, default_value=gui_default,
                             **STYLE_INPUT_ITEM, **callback_options)

    elif dtype == bool or 'action' in params:
        if 'action' in params:
            if params['action'] == 'store_true':
                default = False
            elif params['action'] == 'store_false':
                default = True
            else:
                raise ValueError("ACTION {} NOT MANAGED YET"
                                 .format(params['action']))

        # Could add a checkbox, but could not make it beautiful.
        tmp_default = str(default) if default is not None else None
        item = dpg.add_combo(['True', 'False'], default_value=tmp_default,
                             tag=arg_name, **STYLE_INPUT_ITEM,
                             **callback_options)

    elif dtype == str:
        item = dpg.add_input_text(
            tag=arg_name, default_value=change_none_to_gui_default(default, str),
            **STYLE_INPUT_ITEM, **callback_options)

    elif dtype == int:
        item = dpg.add_input_int(
            tag=arg_name, default_value=change_none_to_gui_default(default, int),
            **STYLE_INPUT_ITEM, **callback_options)

    elif dtype == float:
        item = dpg.add_input_float(
            tag=arg_name, default_value=change_none_to_gui_default(default, float),
            format='%.7f', **STYLE_INPUT_ITEM, **callback_options)

    else:
        item = dpg.add_text("NOT MANAGED YET TYPE {} FOR ARG {}: "
                            .format(dtype, arg_name), tag=arg_name,
                            **STYLE_INPUT_ITEM, **callback_options)

    if default is None:
        if required:
            dpg.bind_item_theme(item, get_required_theme())
        else:
            dpg.bind_item_theme(item, get_none_theme())
    else:
        dpg.bind_item_theme(item, get_non_modified_theme())

    return dtype, default
