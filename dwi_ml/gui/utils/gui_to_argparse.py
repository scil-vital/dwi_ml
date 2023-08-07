# -*- coding: utf-8 -*-
from typing import Dict

import dearpygui.dearpygui as dpg

from dwi_ml.gui.utils.argparse_input_types import CHECKBOX_SUFFIX
from dwi_ml.gui.utils.gui_popup_message import show_infobox


def get_one_value_verify_ok(arg_name):
    required = arg_name[0] != '-'
    item_type = dpg.get_item_type(arg_name)

    # 1. Checking it this is a file dialog
    if item_type == 'mvAppItemType::mvFileDialog':
        raise NotImplementedError(
            "Implementation error. Please use input group from file dialog "
            "rather than accessing file dialog value directly.")

    # 2. Verifying associated checkbox.
    # Will return None if it does not exist.
    checked = dpg.get_value(arg_name + CHECKBOX_SUFFIX)
    value = dpg.get_value(arg_name)
    if value == '':
        # GUI default value.
        value = None
    if checked:
        # We use the default value.
        value = None

    return value, required


def get_all_values(groups: Dict[str, Dict], split_every=50):

    values = ['    ']
    for group_name, group_args in groups.items():
        for tmp_arg_name, params in group_args.items():
            if split_every is not None and len(values[-1]) > split_every:
                values[-1] += "  \\"
                values.append('    ')

            if 'mutually_exclusive_group' in tmp_arg_name:
                arg_name = None
                arg_value = None
                required = False
                for sub_arg_name, sub_params in params.items():
                    tmp_value, required = get_one_value_verify_ok(sub_arg_name)
                    assert not required
                    if tmp_value is not None:
                        # Two mutually exclusive values should not be possible
                        # to choose together.
                        assert arg_value is None

                        arg_name = sub_arg_name
                        arg_value = tmp_value
            else:
                arg_name = tmp_arg_name
                arg_value, required = get_one_value_verify_ok(tmp_arg_name)

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
