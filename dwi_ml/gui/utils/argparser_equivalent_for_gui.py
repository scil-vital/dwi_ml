# -*- coding: utf-8 -*-
import dearpygui.dearpygui as dpg

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


def _set_to_default_callback(sender, app_data, user_data):
    default, dtype = user_data
    if app_data:  # i.e. == True
        nb_char = len('_default_checkbox')
        dpg_item_name = sender[:-nb_char]
        dpg.set_value(dpg_item_name, _manage_visual_default(default, dtype))


def _log_value_and_remove_check(sender, _, __):
    dpg.set_value(sender + '_default_checkbox', False)


def _log_value_exclusive_group(sender, _, elements_in_group):
    raise NotImplementedError

def _add_input_item_based_on_type(arg_name, params, item_options):
    default = None if 'default' not in params else params['default']
    dtype = params['type'] if 'type' in params else str

    if 'choices' in params:
        choices = list(params['choices'])
        default = choices[0] if default is None else default
        dpg.add_combo(choices, tag=arg_name, default_value=default,
                      **style_input_item, **item_options)

    elif 'action' in params:
        if params['action'] == 'store_true':
            # Could add a checkbox, but could not make it beautiful.
            dpg.add_combo(['True', 'False'], default_value='False',
                          tag=arg_name, **style_input_item, **item_options)
        else:
            raise NotImplementedError("NOT READY FOR TYPE TYPE: action: {}"
                                      .format(params['action']))

    elif dtype == str:
        dpg.add_input_text(tag=arg_name,
                           default_value=_manage_visual_default(default, str),
                           **style_input_item, **item_options)

    elif dtype == int:
        dpg.add_input_int(tag=arg_name,
                          default_value=_manage_visual_default(default, int),
                          **style_input_item, **item_options)

    elif dtype == float:
        dpg.add_input_float(tag=arg_name, format= '%.7f',
                            default_value=_manage_visual_default(default, float),
                            **style_input_item, **item_options)

    else:
        dpg.add_text("NOT MANAGED YET TYPE {} FOR ARG {}: "
                     .format(dtype, arg_name), tag=arg_name,
                     **style_input_item, **item_options)

    return dtype


def _add_item_to_group(arg_name, params, required):
    if arg_name == 'exclusive_group':
        raise NotImplementedError
        with dpg.tree_node(label="Select one", default_open=True):
            item_options = {'callback': _log_value_exclusive_group,
                            'user_data': group}
            set_to_default_options = {'callback': _set_to_default_callback}
            for sub_item, sub_params in params.items():
                required = sub_item[0] != '-'
                _add_item_to_gui(sub_item, sub_params, required)
    else:
        # No group.
        if not required:
            # User modified optional value? Else keep default.
            # Default value? Else, we need to manage "None".
            item_options = {'callback': _log_value_and_remove_check}
            set_to_default_options = {'callback': _set_to_default_callback}
            other_user_data = None
        else:
            item_options = {}
            set_to_default_options = {}
            other_user_data = None

        _add_item_to_gui(arg_name, params, required, item_options,
                         set_to_default_options)


def _add_item_to_gui(arg_name, params, required, item_options,
                     set_to_default_option):
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
            dtype = _add_input_item_based_on_type(arg_name, params, item_options)

            # 3. Ignore checkbox.
            if not required:
                default = None if 'default' not in params else params['default']
                dpg.add_checkbox(label='Set to default: {}'.format(default),
                                 default_value=True,
                                 tag=arg_name + '_default_checkbox',
                                 user_data = (default, dtype, other_user_data),
                                 **set_to_default_option)

    # 4. (Below: Help)
    dpg.add_text(params['help'], **style_help)


def add_args_to_gui(args, section_name=None):
    if section_name is not None:
        dpg.add_text('\n' + section_name)

    all_names = list(args.keys())

    all_mandatory = [n for n in all_names if (n != 'exclusive_group' and
                                              n[0] != '-')]
    all_options = [n for n in all_names if (n == 'exclusive_group' or
                                            n[0] == '-')]
    if len(all_mandatory) > 0:
        with dpg.tree_node(label="Required", default_open=False):
            for arg_name in all_mandatory:
                _add_item_to_group(arg_name, args[arg_name], required=True)

    if len(all_options) > 0:
        with dpg.tree_node(label="Options", default_open=False):
            for arg_name in all_options:
                _add_item_to_group(arg_name, args[arg_name], required=False)
