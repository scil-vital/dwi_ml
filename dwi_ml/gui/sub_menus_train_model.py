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


def _log(sender, app_data, user_data):
    print(f"sender: {sender}, \t app_data: {app_data}, \t user_data: {user_data}")
    print("Value: ", dpg.get_value(sender))


style_input_item = {
    'indent': 400,
    'width': 400,
}
style_input_item_show_log = {
    'indent': 400,
    'width': 400,
    'callback': _log
}
style_help = {
    'indent': 100,
    'color': (151, 151, 151, 255)
}


def file_dialog_ok_callback(sender, app_data):
    assert_single_choice_file_dialog(app_data)
    chosen_path = app_data['current_path']
    if sender == "file_dialog_resume_from_checkpoint":
        open_checkpoint_subwindow(chosen_path)

    # else, nothing. We can access the value of the sender later.


def file_dialog_cancel_callback(_, __):
    pass


def add_args_to_gui(args):
    for arg_name, params in args.items():
        with dpg.group(horizontal=True):
            # 1. Argument name
            if arg_name[0:2] == '--':
                dpg.add_text(arg_name + ('.' * 80), indent=40)
            else:
                dpg.add_text(arg_name + ' (**required)' + ('.' * 80), indent=40)

            # 2. Argument value
            if 'type' not in params or params['type'] == str:
                if arg_name == 'experiments_path':
                    # Show dialog
                    dpg.add_button(
                        label='Click here', tag=arg_name, **style_input_item,
                        callback=lambda: dpg.show_item("file_dialog_experiments_path"))
                elif arg_name == 'hdf5_file':
                    dpg.add_button(
                        label='Click here', tag=arg_name, **style_input_item,
                        callback=lambda: dpg.show_item("file_dialog_hdf5_file"))
                else:
                    default = params['default'] if 'default' in params else None
                    # ??? Hint not showing.
                    dpg.add_input_text(hint='Please enter text', tag=arg_name,
                                       **style_input_item_show_log,
                                       default_value=default)
            elif params['type'] == int:
                default = params['default'] if 'default' in params else 0
                dpg.add_input_int(**style_input_item_show_log, tag=arg_name,
                                  default_value=default)
            else:
                dpg.add_text("To be added: " + arg_name, **style_input_item)

        # 3. Help.
        dpg.add_text(params['help'], **style_help)


def add_save_script_outfile():
    dpg.add_text("\n\nScript output", indent=40)
    dpg.add_text("Where to save the output script.", **style_help)
    dpg.add_button(label='Click here', tag='output_script', **style_input_item,
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

            dpg.add_text('\nExperiment:')
            args = get_mandatory_args_training_experiment()
            add_args_to_gui(args)

            dpg.add_text('\nBatch sampler:')
            new_args = get_args_batch_sampler()
            add_args_to_gui(new_args)
            args.update(new_args)

            add_save_script_outfile()
            dpg.add_button(label='Create my script!', indent=900,
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
