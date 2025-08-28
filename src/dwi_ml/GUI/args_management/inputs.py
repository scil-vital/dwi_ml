# -*- coding: utf-8 -*-
from typing import Dict

import dearpygui.dearpygui as dpg

from dwi_ml.GUI.styles.my_styles import INPUTS_WIDTH
from dwi_ml.GUI.styles.theme_inputs import (
    get_required_theme, callback_log_value_change_style_required)


def add_input_item_based_on_type(params: Dict, parent, tag: str,
                                 mendatory=False):
    """
    Add the input. Here, default and const are managed the same way.

    Parameters
    ----------
    params: dict
        The dict of the argparser equivalent.
    parent: str
        The parent
    tag: str
        The tag. (argname_nbX, with X = the nargs number)
    mendatory: bool
        If true, will set it to red as long as it is not set. Else (optional)
        no need to set to red: if not set, the button is not clicked and the
        input space does not show.
    """
    # Test that parent exists
    assert dpg.get_item_alias(parent) is not None, \
        ("Expecting to add item tag={} to parent={}, but parent does not "
         "exist!".format(tag, parent))

    default = params['const'] or params['default']
    dtype = params['type']

    item_options = {
        'tag': tag,
        'width': INPUTS_WIDTH,
        'parent': parent,
    }

    if mendatory:
        item_options['callback'] = callback_log_value_change_style_required,

    if dtype == bool:
        # Currently a dropdown. A checkbox would be more efficient but right
        # now was very ugly. TODO
        tmp_default = str(default) if default is not None else None
        item = dpg.add_combo(['True', 'False'], default_value=tmp_default,
                             **item_options)

    elif params['choices'] is not None:
        choices = list(params['choices'])
        # default is '' if not set. Could also set default to first value?
        item = dpg.add_combo(choices, **item_options)

    elif dtype == str:
        item = dpg.add_input_text(default_value=default or '', **item_options)

    elif dtype == int:
        item = dpg.add_input_int(default_value=default or 0, **item_options)

    elif dtype == float:
        # input_float NEVER gives the right result.
        # Ex: 0.35 gives 0.34999994039
        item = dpg.add_input_double(default_value=default or 0.0,
                                    **item_options)

    else:
        item = dpg.add_text("NOT MANAGED YET TYPE {}"
                            .format(dtype), **item_options)

    if mendatory:
        dpg.bind_item_theme(item, get_required_theme())

    return dtype, default, item
