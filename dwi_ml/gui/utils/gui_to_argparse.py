# -*- coding: utf-8 -*-
import logging
from typing import Dict

import dearpygui.dearpygui as dpg

from dwi_ml.gui.utils.argparse_to_gui import CHECKBOX_SUFFIX
from dwi_ml.gui.utils.file_dialogs import get_file_dialog_value
from dwi_ml.gui.utils.gui_popup_message import show_infobox


def get_one_value_verify_ok(arg_name):
    required = arg_name[0] != '-'

    # 1. Checking it this is a file dialog (SHOULD NOT. INPUT GROUP SHOULD
    # BE PREFERRED)
    info = dpg.get_item_info(arg_name)
    if info['type'] == 'mvAppItemType::mvFileDialog':
        print("? Getting a file dialog value")
        value = get_file_dialog_value(arg_name)
    else:
        # 2. If not required, we have a checkbox. Let's see if it exists.
        checked = dpg.get_value(arg_name + CHECKBOX_SUFFIX)
        value = dpg.get_value(arg_name)
        if value == '':
            value = None
        logging.debug("Getting arg: {}, value: {}".format(arg_name, value))
        logging.debug("    Change for default?: {}".format(checked))

        if checked is None:
            assert required
            # No checkbox = required value.
        else:
            assert not required
            if checked:
                # We use the default value.
                value = None

            # Else: not checked = not using default value.
    return value, required


def get_all_values(groups: Dict[str, Dict], split_every=50):

    values = ['    ']
    for group_name, group_args in groups.items():
        for tmp_arg_name, params in group_args.items():
            if split_every is not None and len(values[-1]) > split_every:
                values[-1] += "  \\"
                values.append('    ')

            if tmp_arg_name == 'mutually_exclusive_group':
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
            if required:
                if arg_value is None:
                    show_infobox("Missing argument",
                                 "Required argument {} was not filled in!"
                                 .format(arg_name))
                    # Exit the call back process
                    return None
                else:
                    values[-1] += '   ' + str(arg_value)
            elif arg_value is not None:
                values[-1] += '   ' + arg_name + ' ' + str(arg_value)

    return values
