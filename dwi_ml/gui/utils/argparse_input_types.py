# -*- coding: utf-8 -*-
from typing import Dict

import dearpygui.dearpygui as dpg

from dwi_ml.gui.utils.my_styles import get_required_theme, \
    get_modified_theme, INDENT_ITEM, WIDTH_ITEM, callback_log_value_change_style


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


def add_input_item_based_on_type(arg_name: str, params: Dict, tag: str = None,
                                 before=None):

    default = None if 'default' not in params else params['default']
    dtype = params['type'] if 'type' in params else str
    tag = tag if tag is not None else arg_name

    item_options = {
        'tag': tag,
        'callback': callback_log_value_change_style,
        'indent': INDENT_ITEM,
        'width': WIDTH_ITEM,
        'before': before
    }

    if 'choices' in params:
        choices = list(params['choices'])
        gui_default = choices[0] if default is None else default
        item = dpg.add_combo(choices, default_value=gui_default,
                             **item_options)

    elif dtype == bool:
        tmp_default = str(default) if default is not None else None
        item = dpg.add_combo(['True', 'False'], default_value=tmp_default,
                             **item_options)

    elif dtype == str:
        item = dpg.add_input_text(
            default_value=change_none_to_gui_default(default, str),
            **item_options)

    elif dtype == int:
        item = dpg.add_input_int(
            default_value=change_none_to_gui_default(default, int),
            **item_options)

    elif dtype == float:
        item = dpg.add_input_float(
            default_value=change_none_to_gui_default(default, float),
            format='%.7f', **item_options)

    else:
        item = dpg.add_text("NOT MANAGED YET TYPE {} FOR ARG {}: "
                            .format(dtype, arg_name), **item_options)

    dpg.bind_item_theme(item, get_required_theme())
