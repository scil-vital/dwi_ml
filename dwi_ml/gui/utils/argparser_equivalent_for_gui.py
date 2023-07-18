# -*- coding: utf-8 -*-
from typing import Dict

import dearpygui.dearpygui as dpg

from dwi_ml.gui.utils.gui_popup_message import show_infobox
from dwi_ml.gui.utils.my_styles import get_none_theme, get_non_modified_theme, \
    get_modified_theme, get_required_theme

style_input_item = {
    'indent': 600,
    'width': 400,
}
style_help = {
    'indent': 100,
    'color': (151, 151, 151, 255)
}
arg_name_indent = 40
NB_DOTS = 150


def _manage_gui_default(default, dtype):
    if default is not None:
        # Assert that default is the right type?
        return default
    else:
        if dtype == str:
            return ''
        elif dtype == int:
            return 0
        elif dtype == float:
            return 0.0
        else:
            raise ValueError("Data type {} not supported yet!".format(dtype))


def _checkbox_set_to_default_callback(sender, app_data, user_data):
    default, dtype, exclusive_group = user_data

    # Value of checkbox is modified automatically.

    # In both cases, changing background color.
    # If setting from unchecked to checked again, also setting value to default.
    nb_char = len('_default_checkbox')
    arg_name = sender[:-nb_char]
    if app_data:  # i.e. == True
        dpg.set_value(arg_name, _manage_gui_default(default, dtype))

        if default is None:
            dpg.bind_item_theme(arg_name, get_none_theme())
        else:
            dpg.bind_item_theme(arg_name, get_non_modified_theme())

    else:
        dpg.bind_item_theme(arg_name, get_modified_theme())

        if exclusive_group is not None:
            # Choosing this one: checking other values.
            for a in exclusive_group:
                if a != arg_name:
                    dpg.set_value(a + '_default_checkbox', True)
                    # to do: set correct default.
                    dpg.bind_item_theme(a, get_non_modified_theme())


def _log_value_and_remove_check(sender, _, __):
    # Changing background color
    dpg.bind_item_theme(sender, get_modified_theme())

    # Removing check.
    dpg.set_value(sender + '_default_checkbox', False)


def _log_value(sender, _, __):
    dpg.bind_item_theme(sender, get_modified_theme())


def _log_value_exclusive_group(sender, _, elements_in_group):
    _log_value_and_remove_check(sender, None, None)

    # Checking other elements (to None, probably)
    for a in elements_in_group:
        if a != sender:
            dpg.set_value(a + '_default_checkbox', True)
            dpg.bind_item_theme(a, get_non_modified_theme())


def _add_input_item_based_on_type(arg_name, params, required,
                                  callback_options=None):
    if callback_options is None:
        callback_options = {}

    default = None if 'default' not in params else params['default']
    dtype = params['type'] if 'type' in params else str

    if 'choices' in params:
        choices = list(params['choices'])
        gui_default = choices[0] if default is None else default
        item = dpg.add_combo(choices, tag=arg_name, default_value=gui_default,
                             **style_input_item, **callback_options)

    elif dtype == bool or 'action' in params:
        if 'action' in params:
            if params['action'] == 'store_true':
                default = False
            elif params['action'] == 'store_false':
                default = True
            else:
                raise ValueError("ACTION {} NOT MANAGED YET"
                                 .format(params['action']))

        # Could add a checkbox, but could not make it beautiful.
        tmp_default = str(default) if default is not None else None
        item = dpg.add_combo(['True', 'False'], default_value=tmp_default,
                             tag=arg_name, **style_input_item,
                             **callback_options)

    elif dtype == str:
        item = dpg.add_input_text(tag=arg_name,
                                  default_value=_manage_gui_default(default, str),
                                  **style_input_item, **callback_options)

    elif dtype == int:
        item = dpg.add_input_int(tag=arg_name,
                                 default_value=_manage_gui_default(default, int),
                                 **style_input_item, **callback_options)

    elif dtype == float:
        item = dpg.add_input_float(tag=arg_name, format='%.7f',
                                   default_value=_manage_gui_default(default, float),
                                   **style_input_item, **callback_options)

    else:
        item = dpg.add_text("NOT MANAGED YET TYPE {} FOR ARG {}: "
                            .format(dtype, arg_name), tag=arg_name,
                            **style_input_item, **callback_options)

    if default is None:
        if required:
            dpg.bind_item_theme(item, get_required_theme())
        else:
            dpg.bind_item_theme(item, get_none_theme())
    else:
        dpg.bind_item_theme(item, get_non_modified_theme())

    return dtype, default


def _add_item_to_group(arg_name, params, existing_file_dialogs, required):
    if arg_name == 'mutually_exclusive_group':
        with dpg.tree_node(label="Select maximum one", default_open=True,
                           indent=40):
            exclusive_args = list(params.keys())
            for sub_item, sub_params in params.items():
                assert sub_item[0] == '-', "Parameter {} is in an exclusive " \
                                           "group, so NOT required, but its " \
                                           "name does not start with '-'" \
                                           .format(sub_item)
                _add_item_to_gui(sub_item, sub_params, existing_file_dialogs,
                                 required=False, exclusive_group=exclusive_args)
    else:
        _add_item_to_gui(arg_name, params, existing_file_dialogs, required)


def _add_item_to_gui(arg_name, params, existing_file_dialogs, required,
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
    if required and exclusive_group is not None:
        raise ValueError("Required elements cannot be in an exclusive group.")

    with dpg.group(horizontal=True):
        # 1. Argument name + metavar
        if 'metavar' in params:
            m = params['metavar']
        elif required or 'action' in params and \
                params['action'] in ['store_true', 'store_false']:
            m = ''
        else:
            m = arg_name.replace('-', '').upper()

        suffix = '   ' + m + ('.' * (NB_DOTS - len(m)))
        dpg.add_text(arg_name + suffix, indent=arg_name_indent)

        # 2. Argument value
        if existing_file_dialogs is None:
            existing_file_dialogs = []
        if arg_name in existing_file_dialogs:
            # Special cases for known arguments that require a file dialog.
            dpg.add_button(
                tag=arg_name + '_button',
                label='Click here to select path', **style_input_item,
                callback=lambda: dpg.show_item(arg_name))
        else:

            # Other cases: depending on type.
            if required:
                item_options = {'callback': _log_value}
            elif exclusive_group is None:
                item_options = {'callback': _log_value_and_remove_check}
            else:
                item_options = {'callback': _log_value_exclusive_group,
                                'user_data': exclusive_group}
            dtype, default = _add_input_item_based_on_type(
                arg_name, params, required, item_options)

            # 3. Ignore checkbox.
            if not required:
                dpg.add_checkbox(label='Set to default: {}'.format(default),
                                 default_value=True,
                                 tag=arg_name + '_default_checkbox',
                                 callback=_checkbox_set_to_default_callback,
                                 user_data=(default, dtype, exclusive_group))

    # 4. (Below: Help)
    dpg.add_text(params['help'], **style_help)


def add_args_groups_to_gui(groups: Dict[str, Dict], existing_file_dialogs=None):
    """
    args: dict of dicts.
        keys are groups. values are dicts of argparser equivalents.
    """
    for group_name, group_args_dict in groups.items():
        dpg.add_text('\n' + group_name + ':')

        # Verifying all arg names in this group
        #  (or sub-args in the case of mutually_exclusive_group)
        all_names = list(group_args_dict.keys())
        all_mandatory = [n for n in all_names if
                         (n != 'mutually_exclusive_group' and n[0] != '-')]
        all_options = [n for n in all_names if
                       (n == 'mutually_exclusive_group' or n[0] == '-')]

        # Separating required and optional
        if len(all_mandatory) > 0:
            with dpg.tree_node(label="Required", default_open=True):
                for arg_name in all_mandatory:
                    _add_item_to_group(arg_name, group_args_dict[arg_name],
                                       existing_file_dialogs, required=True)

        if len(all_options) > 0:
            with dpg.tree_node(label="Options", default_open=False):
                for arg_name in all_options:
                    _add_item_to_group(arg_name, group_args_dict[arg_name],
                                       existing_file_dialogs, required=False)


def _get_one_value_verify_ok(arg_name, params):
    # 1. Checking it this is a file dialog.
    info = dpg.get_item_info(arg_name)
    if info['type'] == 'mvAppItemType::mvFileDialog':
        user_data = dpg.get_item_user_data(arg_name)

        if user_data is None:
            user_data = "TMP FAKE PATH -- WILL SHOW ERROR INFOBOX"
            return user_data
            show_infobox("Missing argument",
                         "Required argument {} was not filled in! "
                         "Please select a path in the file dialog."
                         .format(arg_name))
            return 'stop_process'
        chosen_path = user_data[arg_name]
        return chosen_path

    # 2. If not required, we have a checkbox. Let's see if it exists.
    print("Verifying checkbox for ", arg_name)
    checked = dpg.get_value(arg_name + '_default_checkbox')
    print("Found checkbox value: ", checked)

    value = dpg.get_value(arg_name)
    print("Raw value is", value)
    if checked is None:
        # No checkbox = required value.
        print("REQUIRED ARG.", dpg.get_item_callback(arg_name))
        if value is None:
            return "TMP FAKE VALUE -- WILL SHOW ERROR BOX"
            show_infobox("Missing argument",
                         "Required argument {} was not filled in!"
                         .format(arg_name))
            return 'stop_process'
        return value
    elif checked:
        # We use the default value.
        return None
    else:
        # Not checked = not using default value.
        return arg_name + '  ' + str(value)


def get_all_values(groups: Dict[str, Dict], split_every=50):

    values = ['    ']
    for group_name, group_args in groups.items():
        for arg_name, params in group_args.items():
            if split_every is not None and len(values[-1]) > split_every:
                values[-1] += "  \\"
                values.append('    ')

            if arg_name == 'mutually_exclusive_group':
                chosen_value = None
                for sub_arg_name, sub_params in params.items():
                    tmp_value = _get_one_value_verify_ok(sub_arg_name, params)
                    if tmp_value is not None:
                        if chosen_value is not None:
                            raise ValueError(
                                "TWO MUTUALLY EXCLUSIVE VALUE CHOSEN. SHOULD "
                                "NOT HAPPEN. IMPLEMENTATION ERROR.")
                        chosen_value = tmp_value
                if chosen_value is not None:
                    values[-1] += chosen_value
            else:
                values[-1] = _get_one_value_verify_ok(arg_name, params)

    return values
