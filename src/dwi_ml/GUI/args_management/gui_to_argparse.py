# -*- coding: utf-8 -*-
from typing import Dict

import dearpygui.dearpygui as dpg

from dwi_ml.GUI.args_management.argparse_equivalent import ArgparserForGui
from dwi_ml.GUI.args_management.args_to_gui_one_arg import OPTION_CHECKBOX
from dwi_ml.GUI.utils.gui_popup_message import show_infobox


def get_all_values_as_str(argparser: ArgparserForGui, split_every=50):

    values = ['    ']
    # Let's start with required values.
    for arg_name, params in argparser.required_args_dict.items():
        is_ok, value = get_one_value_if_set(arg_name)
        if not is_ok:
            # Showed a message box. Leaving now.
            return
        print("For required arg '{}': got value '{}'"
              .format(arg_name, value))

    return
    for group_name, group_args in argparser.items():
        for tmp_arg_name, params in group_args.items():
            if split_every is not None and len(values[-1]) > split_every:
                values[-1] += "  \\"
                values.append('    ')

            if 'mutually_exclusive_group' in tmp_arg_name:
                arg_name = None
                arg_value = None
                required = False
                for sub_arg_name, sub_params in params.items():
                    tmp_value, required = get_one_value_if_set(sub_arg_name)
                    assert not required
                    if tmp_value is not None:
                        # Two mutually exclusive values should not be possible
                        # to choose together.
                        assert arg_value is None

                        arg_name = sub_arg_name
                        arg_value = tmp_value
            else:
                arg_name = tmp_arg_name
                arg_value, required = get_one_value_if_set(tmp_arg_name)

            # Ok. Now format to bash script
            if 'action' in params and \
                    params['action'] in ['store_true', 'store_false']:
                # Name only.  Ex: --use_some_option.
                values[-1] += '   ' + str(arg_name)
            elif required:
                if arg_value is None:
                    show_infobox("Missing argument",
                                 "Required argument {} was not filled in!"
                                 .format(arg_name))
                    # Exit the call back process
                    return None
                else:
                    # Value only. Ex: script.py in_file out_file.
                    values[-1] += '   ' + str(arg_value)
            elif arg_value is not None:
                # Name + value. Ex: --optionX value N
                values[-1] += '   ' + arg_name + ' ' + str(arg_value)

    return values


def get_one_value_if_set(arg_name):

    # Verifying if it is optional (has an associated checkbox).
    # Will return None if it does not exist.
    if dpg.does_item_exist(arg_name + OPTION_CHECKBOX):
        is_checked = dpg.get_value(arg_name + OPTION_CHECKBOX)
        if not is_checked:
            # No value defined! Not adding to bash script, will use the default
            # value.
            value = None
        else:
            value = get_value_from_narg_group(arg_name)
    else:
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

    # 1. Checking it this is a file dialog
    if item_type == 'mvAppItemType::mvFileDialog':
        raise NotImplementedError(
            "Implementation error. Please contact developpers with this "
            "information: use input group from file dialog "
            "rather than accessing file dialog value directly!")

    return None
