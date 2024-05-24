# -*- coding: utf-8 -*-
import logging
from typing import Dict

import dearpygui.dearpygui as dpg

from dwi_ml.GUI.args_management.argparse_equivalent import ArgparserForGui
from dwi_ml.GUI.args_management.args_to_gui_one_arg import OPTION_CHECKBOX
from dwi_ml.GUI.utils.gui_popup_message import show_infobox

MAX_LEN_BASH = 100


def get_all_values_as_str(argparser: ArgparserForGui):

    values = ['     ']
    # Let's start with required values.
    logging.debug("Processing required args")
    for arg_name, params in argparser.required_args_dict.items():
        is_ok, is_used, value = get_one_value_if_set(arg_name)
        if not is_ok:
            # Showed a message box. Leaving now.
            return None
        values[-1] += ' {} '.format(value)
        if len(values[-1]) > MAX_LEN_BASH:
            values.append('     ')

    for arg_name, params in argparser.optional_args_dict.items():
        is_ok, is_used, value = get_one_value_if_set(arg_name)
        values[-1] += ' {} {} '.format(arg_name, value)
        if len(values[-1]) > MAX_LEN_BASH:
            values.append('     ')

    return values


def get_one_value_if_set(arg_name):

    # Verifying if it is optional (has an associated checkbox).
    # Will return None if it does not exist.
    if dpg.does_item_exist(arg_name + OPTION_CHECKBOX):
        logging.debug("Verifying optional arg {}".format(arg_name))
        is_checked = dpg.get_value(arg_name + OPTION_CHECKBOX)
        if not is_checked:
            # No value defined! Not adding to bash script, will use the default
            # value.
            value = True
        else:
            value = get_value_from_narg_group(arg_name)
    else:
        logging.debug("Verifying required arg {}".format(arg_name))
        # Required arg.
        value = get_value_from_narg_group(arg_name)

        if value is None or value == '':
            show_infobox("Missing required value",
                         "Value for required argument '{}' is not set!"
                         .format(arg_name))
            return False, None
    return True, value


def get_value_from_narg_group(arg_name):
    item_type = dpg.get_item_type(arg_name)
    logging.debug("   Item type is: {}".format(item_type))

    # 1. Checking it this is a file dialog
    if item_type == 'mvAppItemType::mvFileDialog':
        raise NotImplementedError(
            "Implementation error. Please contact developpers with this "
            "information: use input group from file dialog "
            "rather than accessing file dialog value directly!")
    elif item_type == 'mvAppItemType::mvInputText':
        return dpg.get_value(arg_name)
    else:
        raise NotImplementedError
