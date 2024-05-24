# -*- coding: utf-8 -*-
import logging

import dearpygui.dearpygui as dpg

from dwi_ml.GUI.args_management.argparse_equivalent import ArgparserForGui
from dwi_ml.GUI.args_management.args_to_gui_one_arg import \
    add_one_arg_and_help_to_gui
from dwi_ml.GUI.styles.my_styles import get_my_fonts_dictionary


def add_args_to_gui(gui_parser: ArgparserForGui, known_file_dialogs=None):
    """
    Adds argparsing groups of arguments to the GUI

    Parameters
    ----------
    gui_parser: ArgparserForGui

    known_file_dialogs: dict of str
        keys are arg names that should be shown as file dialogs.
        values are a list: [str, [str]]:
            First str = one of ['unique_file', 'unique_folder']
                (or eventually multi-selection, not implemented yet)
            List of str = accepted extensions (for files only).
        ex: { 'out_filename': ['unique_file', ['.txt']]}
    """
    logging.debug("    GUI.args_management.args_to_gui_all_args: adding args "
                  "to GUI.")
    my_fonts = get_my_fonts_dictionary()

    # 1. Required args:
    title = dpg.add_text('\n Required arguments:')
    dpg.bind_item_font(title, my_fonts['group_title'])
    for argname, argoptions in gui_parser.required_args_dict.items():
        add_one_arg_and_help_to_gui(argname, argoptions, known_file_dialogs)

    # 2. Grouped options
    # At first, we had the idea to add them into a closable section.
    # We didn't like it that much. Removed.
    for sub_parser in gui_parser.groups_of_args:
        # There should not be required args in groups.
        assert len(sub_parser.required_args_dict.keys()) == 0

        # There should not be a subgroup in a group
        assert len(sub_parser.groups_of_args) == 0

        title = dpg.add_text('\n' + sub_parser.description + ':')
        dpg.bind_item_font(title, my_fonts['group_title'])

        _add_options_group_to_gui(sub_parser, known_file_dialogs)

    # 3. Non-grouped args ("Other options")
    title = dpg.add_text('\n Other options:')
    dpg.bind_item_font(title, my_fonts['group_title'])
    _add_options_group_to_gui(gui_parser, known_file_dialogs)


def _add_options_group_to_gui(gui_parser, known_file_dialogs):
    """
    Adds arguments under the group's title.
    """
    # 2. "normal" optional args:
    for argname, argoptions in gui_parser.optional_args_dict.items():
        add_one_arg_and_help_to_gui(argname, argoptions, known_file_dialogs)

    # 2. Mutually exclusive args.
    # (They will always be listed last, contrary to Argparser, but it doesn't
    #  really matter)
    for group in gui_parser.exclusive_groups:
        exclusive_list = list(group.arguments_list.keys())

        if group.required:
            raise NotImplementedError("TODO")
        with dpg.tree_node(label="Select maximum one", default_open=True,
                           indent=0):
            for argname, params in group.arguments_list.items():
                if argname[0] != '-':
                    raise NotImplementedError(
                        "Implementation error. Please contact our developers "
                        "with this information: Parameter {} is in an "
                        "exclusive group, so NOT required, but its name does "
                        "not start with '-'".format(argname))
                add_one_arg_and_help_to_gui(
                    argname, params, known_file_dialogs,
                    exclusive_list=exclusive_list)
