# -*- coding: utf-8 -*-
from typing import Optional

import dearpygui.dearpygui as dpg

from dwi_ml.gui.utils.my_styles import get_required_theme, \
    get_none_theme, get_non_modified_theme, get_modified_theme, INDENT_ITEM, \
    WIDTH_ITEM, INDENT_MORE_NARG

CHECKBOX_SUFFIX = '_default_checkbox'


def _checkbox_set_to_default_callback(sender, app_data, user_data):
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
    arg_defaults, dtype, exclusive_group = user_data

    for input_name, default in arg_defaults.items():
        # In most cases, only one loop, for the assiociated item.
        # Special case with nargs.
        if app_data:  # i.e. == True
            # If setting from unchecked to checked again, also setting value to
            # default.
            dpg.set_value(input_name, change_none_to_gui_default(default, dtype))

            # Changing background color.
            if default is None:
                dpg.bind_item_theme(input_name, get_none_theme())
            else:
                dpg.bind_item_theme(input_name, get_non_modified_theme())

        else:
            # Arg value is modified by the user.
            # Changing background color.
            dpg.bind_item_theme(input_name, get_modified_theme())

    # Verifying that other values in the exclusive groups are not
    # modified. If so, unselecting them.
    if exclusive_group is not None:
        for a in exclusive_group:
            if a != input_name:
                dpg.set_value(a + CHECKBOX_SUFFIX, True)
                dpg.bind_item_theme(a, get_non_modified_theme())


def change_none_to_gui_default(value, dtype):
    """
    Usage: when a default value for a GUI input would be None (which is not
    possible for the GUI), change its value to a 'normal' default value, based
    on the data type.
    """
    if value is not None:
        # Assert that default is the right type?
        return value
    else:
        if dtype == str:
            return ''
        elif dtype == int:
            return 0
        elif dtype == float:
            return 0.0
        else:
            raise ValueError("Data type {}' default not supported yet!"
                             .format(dtype))


def _log_value_verify_exclusive_group_and_checkbox(sender, app_data, user_data):
    """
    Callback intended for input items belonging to a mutually exclusive group.
    By default, they are always optional and thus are associated to a default
    checkbox.
    """
    elements_in_group, has_check = user_data

    # Changing background color
    if str(dpg.get_item_type(sender)) == 'mvAppItemType::mvCheckbox':
        # Checkbox. If unset: unmodified
        if app_data:
            dpg.bind_item_theme(sender, get_modified_theme())
        else:
            dpg.bind_item_theme(sender, get_non_modified_theme())
    else:
        dpg.bind_item_theme(sender, get_modified_theme())

    # Removing check.
    if has_check:
        dpg.set_value(sender + CHECKBOX_SUFFIX, False)

    # Re-checking (to None) other elements.
    if elements_in_group is not None:
        for a in elements_in_group:
            # If a == sender: keep value
            # If a in sender (ex: value in value_narg1): same narg group
            if a not in sender:
                # Other elements are always optional, but either they have a
                # checkbox, or they are themselves a checkbox (ex, 'store_true')
                if dpg.does_item_exist(a + CHECKBOX_SUFFIX):
                    dpg.set_value(a + CHECKBOX_SUFFIX, True)
                else:
                    # store_true, store_false:
                    dpg.set_value(a, False)
                dpg.bind_item_theme(a, get_non_modified_theme())


def _add_input_item(arg_name: str, params, required: bool,
                    exclusive_group: Optional[list],
                    has_check: bool, before=None):

    default = None if 'default' not in params else params['default']
    dtype = params['type'] if 'type' in params else str

    item_options = {
        'tag': arg_name,
        'callback': _log_value_verify_exclusive_group_and_checkbox,
        'user_data': (exclusive_group, has_check),
        'indent': INDENT_ITEM
    }
    if before is not None:
        item_options['before'] = before

    item_options_no_width = item_options.copy()
    item_options['width'] = WIDTH_ITEM

    if 'choices' in params:
        choices = list(params['choices'])
        gui_default = choices[0] if default is None else default
        item = dpg.add_combo(choices, default_value=gui_default, **item_options)

    elif 'action' in params:
        if params['action'] in ['store_true', 'store_false']:
            default = False
            item = dpg.add_checkbox(default_value=default,
                                    **item_options_no_width)
        else:
            raise NotImplementedError("ACTION {} NOT MANAGED YET"
                                      .format(params['action']))
    elif dtype == bool:
        tmp_default = str(default) if default is not None else None
        item = dpg.add_combo(['True', 'False'], default_value=tmp_default,
                             **item_options)

    elif dtype == str:
        item = dpg.add_input_text(
            default_value=change_none_to_gui_default(default, str),
            **item_options)

    elif dtype == int:
        item = dpg.add_input_int(
            default_value=change_none_to_gui_default(default, int),
            **item_options)

    elif dtype == float:
        item = dpg.add_input_float(
            default_value=change_none_to_gui_default(default, float),
            format='%.7f', **item_options)

    else:
        item = dpg.add_text("NOT MANAGED YET TYPE {} FOR ARG {}: "
                            .format(dtype, arg_name), **item_options)

    return dtype, default, item


def _callback_add_new_narg(_, __, user_data):
    """
    Callback to add a new input.
    user_data: arg_info: dict, {arg_name: nb_nargs},
               other_params: tuple, (params, for, add_input_item)
    """
    arg_info, other_params = user_data
    arg_name = list(arg_info.keys())[0]
    nb_nargs = arg_info[arg_name]

    # tag = nb_nargs + 1 - 1 (for python index)
    _, _, item = _add_input_item(arg_name + 'narg' + str(nb_nargs),
                                 *other_params, before=arg_name + 'help')
    dpg.bind_item_theme(item, get_non_modified_theme())
    arg_info[arg_name] += 1


def _callback_remove_new_nargs(_, __, user_data):
    arg_info, nmin = user_data
    arg_name = list(arg_info.keys())[0]
    nb_nargs = arg_info[arg_name]

    for n in range(nmin, nb_nargs):
        dpg.delete_item(arg_name + 'narg' + str(n))

    arg_info[arg_name] += nmin


def add_input_item_based_on_type(arg_name: str, params, required: bool,
                                 exclusive_group: Optional[list]):
    """
    Add an input item based on datatype:
    - A dropdown list of choices
    - Boolean value: currently a dropdown with True/False. Could be a checkbox,
        was not as nice visually.
    - Int/float value: A box with value and arrows on the side to
        increase/decrease chosen input.
    - Str: A simple input box.

    Parameters
    ----------
    arg_name: str
        Will be the tag of the input
    params: dict.
        We will verify if it contains key 'choices', and key 'action'
        (store_true, store_false), and key 'dtype' (else the data type is str).
    required: bool
        Modifying item theme based on if it is required.
    exclusive_group: list
        List of other inter-related args.
    """
    # Other cases: depending on type.
    has_check = True
    if required:
        has_check = False
    elif 'action' in params and \
            params['action'] in ['store_true', 'store_false']:
        has_check = False

    if 'nargs' in params:
        nargs = params['nargs']
        if isinstance(nargs, int):
            raise NotImplementedError
            # User must enter n values.
            default = [None] * nargs
            for n in range(nargs):
                has_check = True if n == 0 else False
                tag = arg_name + 'narg' + str(n)
                dtype, default[n], item = _add_input_item(
                    tag, params, required, exclusive_group, has_check)
            has_check = True
        elif nargs == '?':  # i.e. either default or one value. Uses const.
            # For us, it's the same one value with a 'default'.
            assert 'const' in params
            params['default'] = params['const']
            dtype, default, item = _add_input_item(
                arg_name, params, required, exclusive_group, has_check)
        else:
            if nargs == '*':  # Between 0 and inf nb.
                nmin = 0
                default = None
                dtype = None
                item = None
            else:  # Between 1 and inf nb.
                tag = arg_name + 'narg0'
                dtype, default, item = _add_input_item(
                    tag, params, required, exclusive_group, has_check)
                nmin = 1

            has_check = False  # Managed differently
            dict_nb = {arg_name: nmin}
            # Instead a button to add more params
            other_params = (params, required, exclusive_group, has_check)
            dpg.add_button(label="Add more values",
                           callback=_callback_add_new_narg,
                           user_data=(dict_nb, other_params))
            dpg.add_separator()
            # And a button to remove params
            dpg.add_button(label="Remove all values",
                           callback=_callback_remove_new_nargs,
                           user_data=(dict_nb, nmin))
    else:
        dtype, default, item = _add_input_item(
            arg_name, params, required, exclusive_group, has_check)

    if item is not None:
        if default is None:
            if required:
                dpg.bind_item_theme(item, get_required_theme())
            else:
                dpg.bind_item_theme(item, get_none_theme())
        else:
            dpg.bind_item_theme(item, get_non_modified_theme())

    if has_check:
        dpg.add_checkbox(label='Set to default: {}'.format(default),
                         default_value=True,
                         tag=arg_name + '_default_checkbox',
                         callback=_checkbox_set_to_default_callback,
                         user_data=(arg_name, default, dtype, exclusive_group))
