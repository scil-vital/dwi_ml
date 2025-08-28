# -*- coding: utf-8 -*-
"""
Organization of a single argument:

Each argument is one of:
    - A file dialog: input text on the left, button on the right.
        ** The input text associated to the file dialog has the tag=argname.
    - A nargs group, consisting of:
        - If the number of nargs is not fixed: a button to add or remove nargs.
        -

HELP_AND_INPUT_GROUP:
-------------------------------------------------------------------------------
|  HELP_TABLE:                       INPUTS_AND_OPTIONS_TABLE:                |
|  --------------------------------- ---------------------------------------- |
|  | Row 1 = argname + metavar.... | | Column1 = INPUTS | Column2 = OPTIONS   |
|  | ------------------------------| |          narg#0  |    [] Click to set ||
|  | Row 2 = help, help, help      | |          narg#1  |    [] Add narg     ||
|  |        help, help, help       | |          narg#2  |    [] Less narg    ||
|  |        help, help, help       | |          narg#2  |                    ||
|  ------------------------------- | |---------------------- ----------------||
-------------------------------------------------------------------------------
"""
import logging
from typing import Dict, List

import dearpygui.dearpygui as dpg
import numpy as np

from dwi_ml.GUI.args_management.tags_and_texts import HELP_AND_INPUT_GROUP, \
    HELP_TABLE, INPUTS_AND_OPTIONS_TABLE, INPUTS_CELL, OPTION_DEFAULT_TEXTBOX, \
    OPTIONS_CELL, OPTION_CHECKBOX, ADD_NARGS_BUTTON, REMOVE_NARGS_BUTTON, \
    NARGS_GROUP, TEXT_SELECTED, TEXT_UNSELECTED
from dwi_ml.GUI.utils.file_dialogs import add_file_dialog_input_group
from dwi_ml.GUI.args_management.inputs import add_input_item_based_on_type
from dwi_ml.GUI.styles.my_styles import \
    (NB_DOTS, INDENT_ARGPARSE_NAME, STYLE_ARGPARSE_HELP, INPUTS_WIDTH,
     HELP_WIDTH, OPTIONS_WIDTH, get_my_fonts_dictionary)
from dwi_ml.GUI.styles.theme_inputs import get_required_theme


def add_one_arg_and_help_to_gui(
        argname: str, params: Dict, known_file_dialogs: Dict = None,
        exclusive_list: List = None):
    """
    Prepare one arg, as described at the top of this file.
    """
    logging.debug("GUI.args_to_gui_one_arg: add_one_arg_and_help_to_gui "
                  "(TOP METHOD FOR ARG: {}!)".format(argname))

    # default known_file_dialogs is an empty dict, if None
    known_file_dialogs = known_file_dialogs or {}

    # Verify if optional
    optional = True if argname[0] == '-' else False

    # MAIN ITEM GROUP
    with dpg.group(horizontal=True, tag=argname + HELP_AND_INPUT_GROUP):

        # LEFT SIDE = HELP_TABLE
        with dpg.table(header_row=False, width=HELP_WIDTH,
                       tag=argname + HELP_TABLE):
            dpg.add_table_column()  # Single column table.

            # Two rows:
            # 1) Argname + metavar + dots (...................)
            with dpg.table_row():
                metavar = params['metavar']
                metavar = '' if metavar is None else metavar
                metavar = '  ' + metavar
                dots = '.' * NB_DOTS  # If too long, hidden out of table. ok.
                dpg.add_text(argname + metavar + '  ' + dots,
                             indent=INDENT_ARGPARSE_NAME)

            # 2) Help explanation
            with dpg.table_row():
                _help = params['help']
                if params['choices'] is not None:
                    _help += "\nChoices: {}".format(params['choices'])
                help_txt = dpg.add_text(_help, **STYLE_ARGPARSE_HELP)

        MY_FONTS = get_my_fonts_dictionary()
        dpg.bind_item_font(help_txt, MY_FONTS['code'])

        # RIGHT SIDE = INPUTS + OPTIONS TABLE
        with dpg.table(header_row=False, width=INPUTS_WIDTH + OPTIONS_WIDTH,
                       tag=argname + INPUTS_AND_OPTIONS_TABLE):
            # Two-columns
            dpg.add_table_column()
            dpg.add_table_column()

            # One row
            with dpg.table_row():
                # First cell will go in column 1.
                with dpg.table_cell(tag=argname + INPUTS_CELL):
                    # (nargs group OR default text OR file dialog)
                    need_to_add_file_dialog = False
                    required_arg_except_file = False

                    if argname in known_file_dialogs.keys():
                        # Waiting for option cell to be created, for the
                        # "Click to select" button.
                        need_to_add_file_dialog = True
                    elif not optional:
                        # Required arg. Expects an input right away.
                        # Waiting for option cell to be created, for the narg
                        # button.
                        required_arg_except_file = True
                    else:
                        # Optional. Writing something here.
                        if (params['action'] in ['store_true', 'store_false']
                                or exclusive_list is not None):
                            dpg.add_text(TEXT_UNSELECTED,
                                         tag=argname + OPTION_DEFAULT_TEXTBOX)
                        else:
                            # Will add nargs, but through checkbox callback
                            # For now, only prints the default value.
                            dpg.add_text('(Default = {})'
                                         .format(params['default']),
                                         tag=argname + OPTION_DEFAULT_TEXTBOX)

                # Second cell will go in column 2
                with dpg.table_cell(tag=argname + OPTIONS_CELL):
                    if optional:
                        # Add a checkbox to allow receiving an input
                        # (checkbox also manages mutually exclusive groups)
                        dpg.add_checkbox(
                            label='Click to set', default_value=False,
                            tag=argname + OPTION_CHECKBOX,
                            callback=__callback_option_checkbox,
                            parent=argname + OPTIONS_CELL,
                            user_data=(argname, params, exclusive_list))
                    elif need_to_add_file_dialog:
                        add_file_dialog_input_group(
                                argname,
                                file_dialog_params=known_file_dialogs[argname],
                                input_parent=argname + INPUTS_CELL,
                                button_parent=argname + OPTIONS_CELL)
                    elif required_arg_except_file:
                        check_nargs_add_input(argname, params)


def _get_nb_nargs(params):
    """Used by check_nargs_add_input"""
    if params['action'] in ['store_true', 'store_false']:
        gui_narg_min = 0
        gui_narg_max = 0
    else:
        nargs = params['nargs']
        gui_narg_min = 1
        gui_narg_max = 1
        if nargs is not None:
            if isinstance(nargs, int):
                gui_narg_min = nargs
                gui_narg_max = nargs
            elif nargs == '?':
                gui_narg_min = 0
                gui_narg_max = 1
            elif nargs == '*':  # Between 0 and inf nb.
                gui_narg_min = 0
                gui_narg_max = np.Inf
            else:  # Between 1 and inf nb.
                assert nargs == '+'
                gui_narg_min = 1
                gui_narg_max = np.Inf
    return gui_narg_min, gui_narg_max


def check_nargs_add_input(argname, params):
    """Used either directly by add_one_arg (for required args) or through
    clicking the optional checkbox (for optional args)."""

    if params['nargs'] == '?':  # This means either 0 or 1 arg.
        if params['const'] is None:
            raise NotImplementedError(
                "Implementation error. Please contact developpers with this "
                "information: nargs='?' but no const value for arg {}?"
                .format(argname))
        _add_nargs_group(argname, 1, params)
    else:
        narg_min, narg_max = _get_nb_nargs(params)
        logging.debug("-- For arg {}, expecting between {} and {} args."
                      .format(argname, narg_min, narg_max))

        if narg_max > narg_min:
            _add_more_less_nargs_buttons(argname, narg_min, narg_max, params)
        _add_nargs_group(argname, narg_min, params)


def _add_more_less_nargs_buttons(argname, narg_min, narg_max, params):
    # Add a button to add more params and another to remove them

    # Mutable value to keep track of nargs:
    current_nb = {'current_nb': narg_min}
    user_data = (current_nb, narg_max, argname, params)

    if not dpg.does_item_exist(argname + OPTIONS_CELL):
        raise NotImplementedError("Wrong implementation. Wanted to add "
                                  "the 'add_more_nargs' button, but "
                                  "parent OPTIONS_CELL does not exist.")
    logging.debug("ADDING MORE NARGS BUTTON")
    dpg.add_button(label="Add more values",
                   parent=argname + OPTIONS_CELL,
                   tag=argname + ADD_NARGS_BUTTON,
                   callback=__callback_add_new_narg,
                   user_data=user_data)

    # Add a button to remove params, but deactivated for now
    # Button only exists if it is possible one day to remove nargs.
    logging.debug("ADDING LESS NARGS BUTTON")
    dpg.add_button(label="Remove values",
                   parent=argname + OPTIONS_CELL,
                   tag=argname + REMOVE_NARGS_BUTTON,
                   callback=__callback_remove_new_nargs,
                   user_data=user_data, enabled=False)


def _add_nargs_group(argname, narg_min, params):
    logging.debug("-- Adding NARGS group")
    with dpg.group(tag=argname + NARGS_GROUP,
                   parent=argname + INPUTS_CELL) as nargs_group:
        if params['action'] in ['store_true', 'store_false']:
            # If becomes selected, the first (and only) narg will be: selected.
            dpg.add_text(TEXT_SELECTED , tag=argname + 'narg0')
        else:
            # Add the minimal number of nargs right now.
            for n in range(narg_min):
                logging.debug("-- Adding narg #{}'s input".format(n))
                tag = argname + 'narg{}'.format(n)  # tags start at 0
                add_input_item_based_on_type(params,
                                             parent=nargs_group, tag=tag)


def __callback_add_new_narg(_, __, user_data):
    """
    Callback given to the input section when nargs is set (one per narg).

    user_data:  current_nb: dict
                    {'current_nb': int}, Number of nargs added so far.
                gui_narg_max: int
                    Max nargs allowed
                argname: str
                    argname
                params: dict
                    Argparser params.
    """
    logging.debug("    GUI.args_to_gui_one_arg: Callback add new arg")
    current_nb, gui_narg_max, argname, params = user_data

    nb_nargs = current_nb['current_nb']  # Number of args already added

    # Add the narg input. Narg id numbers start at 0.
    tag = argname + 'narg{}'.format(nb_nargs - 1)
    parent = argname + INPUTS_CELL
    dtype, default, item = add_input_item_based_on_type(
        params, parent, tag)
    dpg.bind_item_theme(item, get_required_theme())

    # We now have one more narg
    current_nb['current_nb'] += 1

    # Enable the "remove one narg" button.
    dpg.enable_item(argname + REMOVE_NARGS_BUTTON)

    # If reached max nb nargs, disable the "Add one narg" button.
    if current_nb['current_nb'] == gui_narg_max:
        dpg.disable_item(argname + ADD_NARGS_BUTTON)  # or disable(sender)


def __callback_remove_new_nargs(_, __, user_data):
    logging.debug("    GUI.args_to_gui_one_arg: Callback remove new arg")
    arg_info, nmin = user_data
    arg_name = list(arg_info.keys())[0]
    nb_nargs = arg_info[arg_name]

    for n in range(nmin, nb_nargs):
        dpg.delete_item(arg_name + 'narg' + str(n))

    arg_info[arg_name] += nmin


def __callback_option_checkbox(_, app_data, user_data):
    """
    Callback for our checkbox that we place beside optional args.

    Value of the checkbox (True, False) is modified automatically. Here,
    changing the value of the associated input.

    user_data:  argname: str
                params: dict
                exclusive_list: list of other args that cannot be modified
                    at the same time.
    """
    argname, params, exclusive_list = user_data
    logging.debug("    GUI.args_to_gui_one_arg: Callback of checkbox arg {}."
                  .format(argname))

    if app_data:
        # Clicking!

        # Remove the "Default = X" text; will replace by input receiver
        dpg.hide_item(argname + OPTION_DEFAULT_TEXTBOX)

        if not dpg.does_item_exist(argname + NARGS_GROUP):
            # If it's the first call, prepare buttons
            check_nargs_add_input(argname, params)
        else:
            # Else, show group back + nargs buttons
            dpg.show_item(argname + NARGS_GROUP)
            if dpg.does_item_exist(argname + ADD_NARGS_BUTTON):
                dpg.show_item(argname + ADD_NARGS_BUTTON)
                dpg.show_item(argname + REMOVE_NARGS_BUTTON)

        # Verifying that other values in the exclusive groups are not also
        # modified. If so, unselect them.
        if exclusive_list is not None:
            for excl_arg in exclusive_list:
                if excl_arg != argname:
                    dpg.set_value(excl_arg + OPTION_CHECKBOX, False)
                    if dpg.does_item_exist(excl_arg + NARGS_GROUP):
                        dpg.hide_item(excl_arg + NARGS_GROUP)

                    # Showing the default text, which is "not selected"
                    dpg.show_item(excl_arg + OPTION_DEFAULT_TEXTBOX)
                    dpg.set_item_label(excl_arg + OPTION_CHECKBOX,
                                       "Unclick to set")

        # Change checkbox text
        dpg.set_item_label(argname + OPTION_CHECKBOX, "Unclick to unset")

    else:
        # Uncliking! (Changed our mind. Not using anymore)

        # Show back text with "default = "
        if dpg.does_item_exist(argname + OPTION_DEFAULT_TEXTBOX):
            dpg.show_item(argname + OPTION_DEFAULT_TEXTBOX)

        # Remove nargs, and possibly buttons
        dpg.hide_item(argname + NARGS_GROUP)
        if dpg.does_item_exist(argname + ADD_NARGS_BUTTON):
            dpg.hide_item(argname + ADD_NARGS_BUTTON)
            dpg.hide_item(argname + REMOVE_NARGS_BUTTON)

        # Change checkbox text
        dpg.set_item_label(argname + OPTION_CHECKBOX, "Click to set")
