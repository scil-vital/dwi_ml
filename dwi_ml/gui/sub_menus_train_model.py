# -*- coding: utf-8 -*-
import os

import dearpygui.dearpygui as dpg
from dwi_ml.gui.utils.inputs import assert_single_choice_file_dialog

from dwi_ml.gui.utils.my_styles import fixed_window_options, \
    get_my_fonts_dictionary, set_global_theme
from dwi_ml.gui.utils.window import callback_change_window
from dwi_ml.io_utils import verify_checkpoint_exists, verify_which_model_in_path
from dwi_ml.training.utils.batch_samplers import get_args_batch_sampler
from dwi_ml.training.utils.experiment import get_mandatory_args_training_experiment

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


def manage_visual_default(default, dtype):
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
        dpg.set_value(dpg_item_name, manage_visual_default(default, dtype))


def _log_value_remove_check(sender, app_data, user_data):
    dpg.set_value(sender + '_default_checkbox', False)


def file_dialog_ok_callback(sender, app_data):
    assert_single_choice_file_dialog(app_data)
    chosen_path = app_data['current_path']
    if sender == "file_dialog_resume_from_checkpoint":
        open_checkpoint_subwindow(chosen_path)
    # else, nothing. We can access the value of the sender later.


def file_dialog_cancel_callback(_, __):
    pass


def add_args_to_gui(args, section_name=None):
    if section_name is not None:
        dpg.add_text('\nExperiment:')

    all_names = list(args.keys())
    all_mandatory = [n for n in all_names if n[0] != '-']
    all_options = [n for n in all_names if n[0] == '-']

    if len(all_mandatory) > 0:
        with dpg.tree_node(label="Required", default_open=True):
            for arg_name in all_mandatory:
                _add_item_to_gui(arg_name, args[arg_name], required=True)

    if len(all_options) > 0:
        with dpg.tree_node(label="Options", default_open=False):
            for arg_name in all_options:
                _add_item_to_gui(arg_name, args[arg_name], required=False)


def _add_item_to_gui(arg_name, params, required):
    # Default value?
    default_is_none = False
    if 'default' not in params:
        default_is_none = True

    # User clicked on optional?
    if not required:
        item_options = {'callback': _log_value_remove_check}
    else:
        item_options = {}

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
                default = None if default_is_none else params['default']
                dpg.add_checkbox(label='Set to default: {}'.format(default),
                                 default_value=True,
                                 tag=arg_name + '_default_checkbox',
                                 user_data=(default, dtype),
                                 callback=_set_to_default_callback)

    # 4. (Below: Help)
    dpg.add_text(params['help'], **style_help)


def _add_input_item_based_on_type(arg_name, params, item_options):
    default = None if 'default' not in params else params['default']
    dtype = params['type'] if 'type' in params else str

    if 'choices' in params:
        choices = list(params['choices'])
        default = choices[0] if default is None else default
        dpg.add_combo(choices, tag=arg_name, default_value=default,
                      **style_input_item, **item_options)

    elif dtype == str:
        dpg.add_input_text(tag=arg_name,
                           default_value=manage_visual_default(default, str),
                           **style_input_item, **item_options)

    elif dtype == int:
        dpg.add_input_int(tag=arg_name,
                          default_value=manage_visual_default(default, int),
                          **style_input_item, **item_options)

    else:
        dpg.add_text("NOT MANAGED YET TYPE {} FOR ARG {}: "
                     .format(dtype, arg_name), tag=arg_name,
                     **style_input_item, **item_options)

    return dtype


def add_save_script_outfile():
    dpg.add_text('\n\n')
    with dpg.group(horizontal=True):
        dpg.add_text("Script output", indent=arg_name_indent)
        dpg.add_button(label='Click here to choose where to save your file',
                       tag='output_script', indent=300, width=600,
                       callback=lambda: dpg.show_item("file_dialog_output_script"))
    dpg.add_text("\n\n\n\n")


def open_checkpoint_subwindow(chosen_path):
    # toDo: if raises a FileNotFoundError, show a pop-up warning?
    #  Currently prints the warning in terminal.
    checkpoint_path = verify_checkpoint_exists(chosen_path)

    model_dir = os.path.join(checkpoint_path, 'model')
    model_type = verify_which_model_in_path(model_dir)
    print("Model's class: {}".format(model_type))

    if model_type == 'Learn2TrackModel':
        open_l2t_from_checkpoint_window()
    elif model_type == 'OriginalTransformerModel':
        open_tto_from_checkpoint_window()
    elif model_type == 'TransformerSrcAndTgtModel':
        open_ttst_from_checkpoint_window()
    else:
        raise ValueError("This type of model is not managed by our DWIML' GUI.")


def open_l2t_from_checkpoint_window():
    with dpg.window(**fixed_window_options):
        pass


def open_tto_from_checkpoint_window():
    print("Allo TTO")


def open_ttst_from_checkpoint_window():
    print("Allo TTST")


def callback_ok_get_args(sender, app_data, args):
    if sender == 'create_l2t_train_script':
        all_values = {}
        for arg_name in args.keys():
            all_values[arg_name] = dpg.get_value(arg_name)

        print("ALL VALUES: ")
        print(all_values)
        create_l2t_train_script(all_values)


def prepare_and_show_train_l2t_window():
    main_window = dpg.get_active_window()
    dpg.hide_item(main_window)

    if dpg.does_item_exist('train_l2t_window'):
        dpg.set_primary_window('train_l2t_window', True)
        dpg.show_item('train_l2t_window')
    else:
        # Create the Learn2track window.
        my_fonts = get_my_fonts_dictionary()
        with dpg.window(**fixed_window_options,
                        tag='train_l2t_window') as l2t_window:
            dpg.set_primary_window(l2t_window, True)

            dpg.add_button(label='<-- Back', callback=callback_change_window,
                           user_data=(l2t_window, main_window))

            title = dpg.add_text('\nLEARN2TRACK:\n\n')
            dpg.bind_font(my_fonts['default'])
            dpg.bind_item_font(title, my_fonts['main_title'])  # NOT WORKING?

            args = get_mandatory_args_training_experiment()
            add_args_to_gui(args, 'Experiment')

            dpg.add_text('\nBatch sampler:')
            new_args = get_args_batch_sampler()
            add_args_to_gui(new_args)
            args.update(new_args)

            add_save_script_outfile()
            dpg.add_button(label='Create my script!', indent=1000,
                           tag='create_l2t_train_script',
                           callback=callback_ok_get_args,
                           user_data=args, height=50)


def create_l2t_train_script(all_values):
    script = "l2t_train_model.py "
    for arg_name, value in all_values.items():
        if value is not None:
            script += arg_name + ' ' + str(value)
        elif not arg_name[0:2] == '--':
            print("Some required values are not defined!")

    print(script)


def open_tto_window():
    print("aallo tto")


def open_ttst_window():
    print("aallo ttst")
