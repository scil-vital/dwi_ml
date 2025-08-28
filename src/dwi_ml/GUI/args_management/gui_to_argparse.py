# -*- coding: utf-8 -*-
import logging

import dearpygui.dearpygui as dpg

from dwi_ml.GUI.args_management.argparse_equivalent import ArgparserForGui
from dwi_ml.GUI.args_management.tags_and_texts import OPTION_CHECKBOX, \
    NARGS_GROUP, TEXT_SELECTED, TEXT_UNSELECTED
from dwi_ml.GUI.utils.gui_popup_message import show_infobox

MAX_LEN_BASH = 100


def get_all_values_as_str(argparser: ArgparserForGui):
    """
    Get the values associated to each argument in the argparser.

    Merge them all in a single string: positional values first, and then
    optional values with --argname before.

    Parameters
    ----------
    argparser : ArgparserForGui
        The argparser imitation for GUI

    Returns
    -------
    values: list[str]
        The list of values. Each str in the list should be shown on a separate
        line in the final command line.
    """
    values = ['     ']

    # Let's start with required values.
    for argname in argparser.get_all_required_names():
        is_ok, value = get_one_required_value(argname)
        if not is_ok:
            # Showed a message box. Leaving now.
            return None

        # Appending value to latest line
        values[-1] += ' {} '.format(value)

        # New line if we have enough variables here.
        if len(values[-1]) > MAX_LEN_BASH:
            values.append('     ')

    # Now the optional values.
    for argname in argparser.get_all_optional_argnames():
        is_checked, is_selected, value = get_one_optional_value_if_set(argname)

        # Appending name + value to latest line
        if is_checked:
            values[-1] += ' {} '.format(argname)
            if not is_selected:  # store_true or store_false
                values[-1] += ' {} '.format(value)

        # New line if we have enough variables.
        if len(values[-1]) > MAX_LEN_BASH:
            values.append('     ')

    return values


def get_one_required_value(argname):
    """
    Get the value associated with the tag of arg_name.

    Parameters
    ----------
    argname : str
        The name of the argument. Will also be the name of the tag.

    Returns
    -------
    is_ok: bool
        If required but not set, a warning is shown and is_ok = False.
    value: str
        The value associated with the tag of arg_name
    """
    is_ok = True
    _, value = get_value_from_narg_group(argname)

    if value is None or value == '':
        show_infobox("Missing required value",
                     "Value for required argument '{}' is not set!"
                     .format(argname))
        is_ok = False

    return is_ok, '{}'.format(value)


def get_one_optional_value_if_set(argname):
    """
    Get the value associated with the tag of arg_name.

    Parameters
    ----------
    argname : str
        The name of the argument. Will also be the name of the tag.

    Returns
    -------
    is_ok: bool
        If required but not set, a warning has been shown and is_ok = False.
    is_checked: bool
        If optional, is the checkbox checked (should be used)
    is_selected: bool
        If optional and checked, is the value "Selected"?
        If None, the text is neither "Selected" nor "(not selected)", so it
        should be a valid value.
    value: Any or None or ' '
        The value associated with the tag of arg_name, or None if it is an
        optional argument not set, or ' ' if set with no value.
    """
    is_selected = None  # Based on the text (selected), ex, for store_true.
    value = None

    # Verifying if it is optional (has an associated checkbox).
    assert dpg.does_item_exist(argname + OPTION_CHECKBOX)
    is_checked = dpg.get_value(argname + OPTION_CHECKBOX)
    if is_checked:
        is_selected, value = get_value_from_narg_group(argname)
    return is_checked, is_selected, value


def get_value_from_narg_group(argname):
    """
    Returns
    -------
    is_selected: bool
        True if the value is a text with "Selected!" meaning it was store_true
        or store_false, False if it is still written (not selected). Else, (not
        sure the case), None.
    value: Any
        The value associated with the tag of argname.
    """
    is_selected = None
    value = False

    # The only arguments that exist directly are file dialogs. Should be
    # the associated text.
    if dpg.does_item_exist(argname):
        item_type = dpg.get_item_type(argname)
        logging.debug("   Item type for {} is: {}".format(argname, item_type))
        assert item_type == 'mvAppItemType::mvInputText', \
            "Unexpectd implementation erreor!?!?!"
        value = dpg.get_value(argname)

    elif dpg.does_item_exist(argname + NARGS_GROUP):
        # All other arguments are set inside a nargs group

        value = ''
        item=0
        while dpg.does_item_exist(argname + 'narg{}'.format(item)):
            item_value = dpg.get_value(argname + 'narg{}'.format(item))
            if item_value == TEXT_SELECTED:  # store_true or store_false. narg0
                value = None
                is_selected = True
                break
            elif item_value == TEXT_UNSELECTED:
                value = None
                is_selected = False
                break
            else:
                value += '{}  '.format(item_value)
            item += 1
    else:
        print("------------> OTHER??? ", argname)

    return is_selected, value
