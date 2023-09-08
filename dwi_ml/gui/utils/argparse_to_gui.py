# -*- coding: utf-8 -*-
import logging
from typing import Dict, List

import dearpygui.dearpygui as dpg
import numpy as np

from dwi_ml.gui.utils.argparse_input_types import add_input_item_based_on_type
from dwi_ml.gui.utils.file_dialogs import add_file_dialog_input_group
from dwi_ml.gui.utils.my_styles import \
    NB_DOTS, INDENT_ARGPARSE_NAME, STYLE_ARGPARSE_HELP, get_my_fonts_dictionary, \
    get_required_theme, INDENT_SET_IT_OPTION, INDENT_ITEM

CHECKBOX_SUFFIX = '_default_checkbox'
NOT_SET_SUFFIX = '_notSetTxt'
MAIN_DPG_GROUP_SUFFIX = '_horizontal_group'
USED_DPG_GROUP_SUFFIX = '_USAGE_GROUP'
ADD_NARGS_SUFFIX = '_add_narg_button'
REMOVE_NARGS_SUFFIX = '_remove_narg_button'


def _callback_add_new_narg(sender, __, user_data):
    current_nb, gui_narg_max, arg_name, params = user_data
    nb_nargs = current_nb['current_nb']  # Number of args already addded

    # If this is the first narg, enable the "remove one narg" button
    dpg.enable_item(arg_name + REMOVE_NARGS_SUFFIX)

    # Position of the input:
    if nb_nargs > 0:
        # After first arg_number 1.
        position_after = arg_name + 'narg' + str(nb_nargs - 1)
    else:
        # After the button to remove nargs
        position_after = arg_name + REMOVE_NARGS_SUFFIX

    # Add the narg input. Narg id numbers start at 0 = nb_nargs + 1 - 1
    tag = arg_name + 'narg' + str(nb_nargs)
    dtype, default, item = add_input_item_based_on_type(
        arg_name, params, tag, before=arg_name + ADD_NARGS_SUFFIX)
    dpg.bind_item_theme(item, get_required_theme())

    # We now have one more narg
    current_nb['current_nb'] += 1

    # Count starts at 0: verifying + 1
    if current_nb['current_nb'] + 1 == gui_narg_max:
        dpg.disable_item(sender)


def _callback_remove_new_nargs(_, __, user_data):
    arg_info, nmin = user_data
    arg_name = list(arg_info.keys())[0]
    nb_nargs = arg_info[arg_name]

    for n in range(nmin, nb_nargs):
        dpg.delete_item(arg_name + 'narg' + str(n))

    arg_info[arg_name] += nmin


def _add_one_arg_in_usage(arg_name, params):
    """
    Adds an input item based on datatype:
    - A dropdown list of choices
    - Boolean value: currently a dropdown with True/False. Could be a checkbox,
        was not as nice visually.
    - Int/float value: A box with value and arrows on the side to
        increase/decrease chosen input.
    - Str: A simple input box.

    For args with 'nargs': If zero args is allowed, another button to add args.
    """
    if dpg.does_alias_exist(arg_name + USED_DPG_GROUP_SUFFIX):
        # Just reactivate
        dpg.show_item(arg_name + USED_DPG_GROUP_SUFFIX)

    else:
        parent = arg_name + MAIN_DPG_GROUP_SUFFIX

        #with dpg.group(tag=arg_name + USED_DPG_GROUP_SUFFIX, horizontal=True,
        #               parent=arg_name + MAIN_DPG_GROUP_SUFFIX):
        # todO: How to show and add group?? Does not show. Else, create it
        #  at creation of the window but then hide it.
        #  Tried: parent: MAIN_GROUP_SUFFIX
        gui_narg_min = 1
        gui_narg_max = 1

        if 'nargs' in params:
            nargs = params['nargs']
            if isinstance(nargs, int):
                gui_narg_min = nargs
                gui_narg_max = nargs
            elif nargs == '?':
                # For us, const is the same as default.
                assert 'const' in params
                assert 'default' not in params
                params['default'] = params['const']
                gui_narg_min = 0
                gui_narg_max = 1
            elif nargs == '*':  # Between 0 and inf nb.
                gui_narg_min = 0
                gui_narg_max = np.Inf
            else:  # Between 1 and inf nb.
                assert nargs == '+'
                gui_narg_min = 1
                gui_narg_max = np.Inf

        if 'action' in params:
            if params['action'] in ['store_true', 'store_false']:
                gui_narg_min = 0
                gui_narg_max = 0
            else:
                # Ex: append, count, store_const, etc.
                # Usually not used in our scripts.
                raise NotImplementedError("ACTION {} NOT MANAGED YET"
                                          .format(params['action']))

        # Add a button to add more params
        if gui_narg_max > gui_narg_min:
            # Mutable value to keep track of nargs:
            current_nb = {'current_nb': gui_narg_min}
            user_data = (current_nb, gui_narg_max, arg_name, params)

            dpg.add_button(label="Add more values",
                           parent=parent,
                           tag=arg_name + ADD_NARGS_SUFFIX,
                           callback=_callback_add_new_narg,
                           user_data=user_data)

            # Add a button to remove params, but deactivated for now
            dpg.add_button(label="Remove values",
                           parent=parent,
                           tag=arg_name + REMOVE_NARGS_SUFFIX,
                           callback=_callback_remove_new_nargs,
                           user_data=user_data, enabled=False)

        # Add the minimal number of nargs right now
        for n in range(gui_narg_min):
            add_input_item_based_on_type(arg_name, params, parent=parent)


def _callback_make_arg_in_usage(_, app_data, user_data):
    """
    Callback for our checkbox that we place beside optional args.
    Useful mainly if the default value is None, which is not possible in the
    GUI. If the user did not modify the default value, we will ignore that
    option.

    Value of the checkbox (True, False) is modified automatically. Here,
    changing the value of the associated input.

    user_data:  dict{input_name: default}
                data type
                exclusive group: dict of other args that cannot be modified
                    at the same time.
    """
    arg_name, params, exclusive_group = user_data

    if app_data:
        # Using this optional arg.

        # Remove 'Not set' text
        dpg.hide_item(arg_name + NOT_SET_SUFFIX)

        # If it requires an input (i.e. if it's not a 'store_true' / 'store_false'
        # action), add it.
        _add_one_arg_in_usage(arg_name, params)

        # Verifying that other values in the exclusive groups are not also
        # modified. If so, unselect them.
        if exclusive_group is not None:
            for a in exclusive_group:
                if a != arg_name:
                    dpg.set_value(a + CHECKBOX_SUFFIX, False)

    else:
        # Changed our mind. Not using anymore.
        dpg.show_item(arg_name + NOT_SET_SUFFIX)

        if dpg.does_item_exist(arg_name + USED_DPG_GROUP_SUFFIX):
            dpg.hide_item(arg_name + USED_DPG_GROUP_SUFFIX)


def _add_one_arg_and_help_to_gui(
        arg_name: str, params: Dict, known_file_dialogs: Dict,
        required: bool, exclusive_group: List = None):
    """
    Adds an item to the GUI

    Parameters
    ----------
    arg_name: str
        Name of the argument (ex: --some_arg).
    params: dict
        argparser arguments in a dictionary. Ex: nargs, help, type, default...
    known_file_dialogs: dict
        The list of arguments associated to file dialogs.
    required: bool
        Wether or not that argument is required. (Starts with - or not).
        If not required, the GUI always requires a default value. A checkbox
        will be added beside the item to allow choosing (real) default, even
        if it is None.
   exclusive_group: list
        Name of other args that cannot be modified together.
   """
    logging.debug("          {}".format(arg_name))

    if known_file_dialogs is None:
        file_dialog_argnames = []
    else:
        file_dialog_argnames = list(known_file_dialogs.keys())

    if required and exclusive_group is not None:
        raise ValueError("Required elements cannot be in an exclusive group.")

    with dpg.group(horizontal=True,
                   tag=arg_name + MAIN_DPG_GROUP_SUFFIX):
        # 1. Argument name + metavar
        if 'metavar' in params:
            m = '   ' + params['metavar']
        elif required or 'action' in params and \
                params['action'] in ['store_true', 'store_false']:
            m = ''
        else:
            # Not required, no metavar: use argname in capital letters
            m = '   ' + arg_name.replace('-', '').upper()

        # Modifying metavar if 'nargs' is set.
        if 'nargs' in params:
            nargs = params['nargs']
            if isinstance(nargs, int):
                m = (m + ' ') * nargs
            elif nargs == '?':  # i.e. either default or one value. Uses const.
                m = '[' + m + ']'
            elif nargs == '*':  # Between 0 and inf nb.
                m = '[' + m + '... ]'
            elif nargs == '+':  # Between 1 and inf nb.
                m = m + ' [' + m + '... ]'

        # 2. Preparing dots (...................) until the input box.
        # Letters are bigger than dots. Removing more dots than number of
        # letters.
        len_text = len(m) + len(arg_name)
        dots = '.' * (NB_DOTS - int(1.5 * len_text))
        dpg.add_text(arg_name + m + '  ' + dots, indent=INDENT_ARGPARSE_NAME)

        # 3. Argument value ( + checkbox to set to None)
        if arg_name in file_dialog_argnames:
            # Special cases for known arguments that require a file dialog.
            assert 'nargs' not in params
            add_file_dialog_input_group(arg_name,
                                        known_file_dialogs[arg_name])
        elif not required:
            dpg.add_text('Not set...', indent=INDENT_ITEM,
                         tag=arg_name + NOT_SET_SUFFIX)
            dpg.add_checkbox(label='Set it', default_value=False,
                             tag=arg_name + CHECKBOX_SUFFIX,
                             callback=_callback_make_arg_in_usage,
                             user_data=(arg_name, params, exclusive_group),
                             indent=INDENT_SET_IT_OPTION)
        else:
            _add_one_arg_in_usage(arg_name, params)

    # 4. (Below: Help)
    dpg.add_text(params['help'], tag=arg_name + 'help', **STYLE_ARGPARSE_HELP)


def _add_exclusive_subgroup(exclusive_group_dict, known_file_dialogs):
    """
    Adds items that are interconnected. Cannot be modified together. Modifying
    one sets the other(s) back to default.

    Parameters
    ----------
    exclusive_group_dict: dict
        The argument dictionary.
    known_file_dialogs: dict
        The list of arguments associated to file dialogs.
    """
    logging.debug("        --> Adding a mutually exclusive group")
    with dpg.tree_node(label="Select maximum one", default_open=True,
                       indent=40):
        exclusive_args = list(exclusive_group_dict.keys())
        for sub_item, sub_params in exclusive_group_dict.items():
            assert sub_item[0] == '-', \
                "Parameter {} is in an exclusive group, so NOT required, " \
                "but its name does not start with '-'".format(
                    sub_item)
            _add_one_arg_and_help_to_gui(
                sub_item, sub_params, known_file_dialogs,
                required=False, exclusive_group=exclusive_args)


def add_groups_of_args_to_gui(groups: Dict[str, Dict], known_file_dialogs=None):
    """
    Adds argparsing groups of arguments to the GUI

    Parameters
    ----------
    groups: dict of dicts.
        keys are groups. values are dicts of argparser equivalents.
    known_file_dialogs: dict of str
        keys are arg names that should be shown as file dialogs.
        values are a list: [str, [str]]:
            First str = one of ['unique_file', 'unique_folder']
                (or eventually multi-selection, not implemented yet)
            List of str = accepted extensions (for files only).
        ex: { 'out_filename': ['unique_file', ['.txt']]}
    """
    logging.debug("Adding argparsing groups to the GUI:")
    my_fonts = get_my_fonts_dictionary()

    for group_name, group_args_dict in groups.items():
        logging.debug("  {}".format(group_name))

        title = dpg.add_text('\n' + group_name + ':')
        dpg.bind_item_font(title, my_fonts['group_title'])

        # Verifying all arg names in this group
        #  (or sub-args in the case of mutually_exclusive_group)
        all_names = list(group_args_dict.keys())
        for arg_name in all_names:
            if 'mutually_exclusive_group' in arg_name:
                _add_exclusive_subgroup(group_args_dict[arg_name],
                                        known_file_dialogs)
            else:
                _add_one_arg_and_help_to_gui(arg_name, group_args_dict[arg_name],
                                             known_file_dialogs, required=False)
