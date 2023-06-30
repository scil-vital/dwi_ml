# -*- coding: utf-8 -*-
import dearpygui.dearpygui as dpg

from dwi_ml.gui.utils.argparser_equivalent_for_gui import add_args_to_gui, \
    add_save_script_outfilename_item
from dwi_ml.gui.utils.my_styles import fixed_window_options, \
    get_my_fonts_dictionary
from dwi_ml.gui.utils.window import callback_change_window
from dwi_ml.training.utils.batch_samplers import get_args_batch_sampler
from dwi_ml.training.utils.experiment import get_mandatory_args_training_experiment


def callback_ok_get_args_l2t(_, __, args):
    # user_data = args
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

            new_args = get_args_batch_sampler()
            add_args_to_gui(new_args, 'Batch sampler')
            args.update(new_args)

            add_save_script_outfilename_item()
            dpg.add_button(label='Create my script!', indent=1000,
                           tag='create_l2t_train_script',
                           callback=callback_ok_get_args_l2t,
                           user_data=args, height=50)


def create_l2t_train_script(all_values):
    script = "l2t_train_model.py "
    for arg_name, value in all_values.items():
        if value is not None:
            script += arg_name + ' ' + str(value)
        elif not arg_name[0:2] == '--':
            print("Some required values are not defined!")

    print(script)


def open_l2t_from_checkpoint_window():
    with dpg.window(**fixed_window_options):
        pass
