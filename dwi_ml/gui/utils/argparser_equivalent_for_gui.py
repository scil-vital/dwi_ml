# -*- coding: utf-8 -*-
import dearpygui.dearpygui as dpg

from dwi_ml.gui.utils.my_styles import create_non_modified_theme, \
    create_none_theme, create_modified_theme

style_input_item = {
    'indent': 600,
    'width': 400,
}
style_help = {
    'indent': 100,
    'color': (151, 151, 151, 255)
}
arg_name_indent = 40
nb_dots = 150
global_non_modified_theme = None
global_modified_theme = None
global_none_theme = None


def _non_modified_theme() -> int:
    global global_non_modified_theme
    if global_non_modified_theme is None:
        global_non_modified_theme = create_non_modified_theme()
    return global_non_modified_theme


def _none_theme() -> int:
    global global_none_theme
    if global_none_theme is None:
        global_none_theme = create_none_theme()
    return global_none_theme


def _modified_theme() -> int:
    global global_modified_theme
    if global_modified_theme is None:
        global_modified_theme = create_modified_theme()
    return global_modified_theme


def _manage_visual_default(default, dtype):
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

    # If setting from unchecked to checked again, setting value to default.
    nb_char = len('_default_checkbox')
    dpg_item_name = sender[:-nb_char]
    if app_data:  # i.e. == True
        dpg.set_value(dpg_item_name, _manage_visual_default(default, dtype))

        if default is None:
            dpg.bind_item_theme(dpg_item_name, _none_theme())
        else:
            dpg.bind_item_theme(dpg_item_name, _non_modified_theme())

        if exclusive_group is not None:
            # Unchecking other values.
            for other_arg in exclusive_group:
                dpg.set_value(other_arg + '_default_checkbox', False)
    else:
        dpg.bind_item_theme(dpg_item_name, _modified_theme())


def _log_value_and_remove_check(sender, _, __):
    dpg.bind_item_theme(sender, _modified_theme())
    dpg.set_value(sender + '_default_checkbox', False)


def _log_value_exclusive_group(sender, _, elements_in_group):
    raise NotImplementedError


def _add_input_item_based_on_type(arg_name, params, item_options=None):
    if item_options is None:
        item_options = {}

    default = None if 'default' not in params else params['default']
    dtype = params['type'] if 'type' in params else str

    if 'choices' in params:
        choices = list(params['choices'])
        default = choices[0] if default is None else default
        item = dpg.add_combo(choices, tag=arg_name, default_value=default,
                             **style_input_item, **item_options)

    elif 'action' in params:
        if params['action'] == 'store_true':
            # Could add a checkbox, but could not make it beautiful.
            item = dpg.add_combo(['True', 'False'], default_value='False',
                                 tag=arg_name, **style_input_item, **item_options)
        else:
            raise NotImplementedError("NOT READY FOR TYPE TYPE: action: {}"
                                      .format(params['action']))

    elif dtype == str:
        item = dpg.add_input_text(tag=arg_name,
                                  default_value=_manage_visual_default(default, str),
                                  **style_input_item, **item_options)

    elif dtype == int:
        item = dpg.add_input_int(tag=arg_name,
                                 default_value=_manage_visual_default(default, int),
                                 **style_input_item, **item_options)

    elif dtype == float:
        item = dpg.add_input_float(tag=arg_name, format= '%.7f',
                                   default_value=_manage_visual_default(default, float),
                                   **style_input_item, **item_options)

    else:
        item = dpg.add_text("NOT MANAGED YET TYPE {} FOR ARG {}: "
                            .format(dtype, arg_name), tag=arg_name,
                            **style_input_item, **item_options)

    if default is None:
        global global_none_theme
        if global_none_theme is None:
            global_none_theme = create_none_theme()
        dpg.bind_item_theme(item, _none_theme())
    else:
        dpg.bind_item_theme(item, _non_modified_theme())

    return dtype


def _add_item_to_group(arg_name, params, required):
    if arg_name == 'exclusive_group':
        with dpg.tree_node(label="Select one", default_open=True):
            # toDo: verify required group.
            group = list(params.keys())
            for sub_item, sub_params in params.items():
                assert sub_item[0] == '-', "Parameter {} is in an exclusive " \
                                           "group, so NOT required, but its " \
                                           "name does not start with '-'" \
                                           .format(sub_item)
                _add_item_to_gui(sub_item, sub_params, required=False,
                                 exclusive_group=group)
    else:
        _add_item_to_gui(arg_name, params, required)


def _add_item_to_gui(arg_name, params, required, exclusive_group=None):
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
        # 1. Argument name
        dpg.add_text(arg_name + ('.' * nb_dots), indent=arg_name_indent)

        # 2. Argument value
        # Special cases for experiments_path and hdf5_file to open file dialogs.
        if arg_name == 'experiments_path':
            dpg.add_button(
                label='Click here to select path',
                tag=arg_name, **style_input_item,
                callback=lambda: dpg.show_item("file_dialog_experiments_path"))
        elif arg_name == 'hdf5_file':
            dpg.add_button(
                label='Click here to select file',
                tag=arg_name, **style_input_item,
                callback=lambda: dpg.show_item("file_dialog_hdf5_file"))
        else:
            if exclusive_group is None:
                item_options = {'callback': _log_value_and_remove_check}
            else:
                item_options = {'callback': _log_value_exclusive_group,
                                'user_data': exclusive_group}
            dtype = _add_input_item_based_on_type(arg_name, params, item_options)

            # 3. Ignore checkbox.
            if not required:
                default = None if 'default' not in params else params['default']
                dpg.add_checkbox(label='Set to default: {}'.format(default),
                                 default_value=True,
                                 tag=arg_name + '_default_checkbox',
                                 callback=_checkbox_set_to_default_callback,
                                 user_data=(default, dtype, exclusive_group))

    # 4. (Below: Help)
    dpg.add_text(params['help'], **style_help)


def add_args_to_gui(args):
    """
    args: dict of dicts.
        keys are groups. values are dicts of argparser equivalents.
    """
    for group, group_args in args.items():
        dpg.add_text('\n' + group + ':')

        all_names = list(group_args.keys())

        all_mandatory = [n for n in all_names if (n != 'exclusive_group' and
                                                  n[0] != '-')]
        all_options = [n for n in all_names if (n == 'exclusive_group' or
                                                n[0] == '-')]
        if len(all_mandatory) > 0:
            with dpg.tree_node(label="Required", default_open=False):
                for arg_name in all_mandatory:
                    _add_item_to_group(arg_name, group_args[arg_name],
                                       required=True)

        if len(all_options) > 0:
            with dpg.tree_node(label="Options", default_open=False):
                for arg_name in all_options:
                    _add_item_to_group(arg_name, group_args[arg_name],
                                       required=False)
