# -*- coding: utf-8 -*-
from typing import Dict

import dearpygui.dearpygui as dpg

from dwi_ml.gui.utils.file_dialogs import add_file_dialog_input_group
from dwi_ml.gui.utils.argparse_input_types import add_input_item_based_on_type
from dwi_ml.gui.utils.my_styles import \
    NB_DOTS, INDENT_ARGPARSE_NAME, STYLE_ARGPARSE_HELP


def _add_exclusive_group(exclusive_group_dict, known_file_dialogs):
    with dpg.tree_node(label="Select maximum one", default_open=True,
                       indent=40):
        exclusive_args = list(exclusive_group_dict.keys())
        for sub_item, sub_params in exclusive_group_dict.items():
            assert sub_item[0] == '-', \
                "Parameter {} is in an exclusive group, so NOT required, " \
                "but its name does not start with '-'".format(
                    sub_item)
            _add_item_to_gui(sub_item, sub_params,
                             known_file_dialogs,
                             required=False,
                             exclusive_group=exclusive_args)


def _add_item_to_gui(arg_name, params, known_file_dialogs, required,
                     exclusive_group=None):
    """
    arg_name: str
        Name of the argument (ex: --some_arg).
    params: dict
        argparser arguments in a dictionary. Ex: nargs, help, type, default...
    required: bool
        Wether or not that argument is required. (Starts with - or not).
        If not required, the GUI always requires a default value. A checkbox
        will be added beside the item to allow choosing (real) default, even
        if it is None.
   exclusive_group: list
        Name of other args that cannot be modified together.
   """
    if known_file_dialogs is None:
        file_dialog_argnames = []
    else:
        file_dialog_argnames = list(known_file_dialogs.keys())

    if required and exclusive_group is not None:
        raise ValueError("Required elements cannot be in an exclusive group.")

    with dpg.group(horizontal=True):
        # 1. Argument name + metavar
        if 'metavar' in params:
            m = '   ' + params['metavar']
        elif required or 'action' in params and \
                params['action'] in ['store_true', 'store_false']:
            m = ''
        else:
            # Not required, no metavar: use argname in capital letters
            m = '   ' + arg_name.replace('-', '').upper()

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

        # Letters are bigger than dots. Removing more dots than number of
        # letters.
        len_text = len(m) + len(arg_name)
        dots = m + ('.' * (NB_DOTS - int(1.5 * len_text)))
        dpg.add_text(arg_name + dots, indent=INDENT_ARGPARSE_NAME)

        # 2. Argument value ( + checkbox to set to None)
        if arg_name in file_dialog_argnames:
            # Special cases for known arguments that require a file dialog.
            assert 'nargs' not in params
            add_file_dialog_input_group(arg_name,
                                        known_file_dialogs[arg_name])
        else:
            add_input_item_based_on_type(
                arg_name, params, required, exclusive_group)

    # 4. (Below: Help)
    dpg.add_text(params['help'], tag=arg_name + 'help', **STYLE_ARGPARSE_HELP)


def add_args_groups_to_gui(groups: Dict[str, Dict], known_file_dialogs=None):
    """
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
    for group_name, group_args_dict in groups.items():
        dpg.add_text('\n' + group_name + ':')

        # Verifying all arg names in this group
        #  (or sub-args in the case of mutually_exclusive_group)
        all_names = list(group_args_dict.keys())
        options = [n for n in all_names if
                   n[0] == '-' or 'mutually_exclusive_group' in n]
        required = list(set(all_names).difference(set(options)))

        # Separating required and optional
        if len(required) > 0:
            with dpg.tree_node(label="Required", default_open=True):
                for arg_name in required:
                    _add_item_to_gui(arg_name, group_args_dict[arg_name],
                                     known_file_dialogs, required=True)

        if len(options) > 0:
            with dpg.tree_node(label="Options", default_open=False):
                for arg_name in options:
                    if 'mutually_exclusive_group' in arg_name:
                        _add_exclusive_group(group_args_dict[arg_name],
                                             known_file_dialogs)
                    else:
                        _add_item_to_gui(arg_name, group_args_dict[arg_name],
                                         known_file_dialogs, required=False)
