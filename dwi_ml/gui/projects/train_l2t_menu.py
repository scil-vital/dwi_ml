# -*- coding: utf-8 -*-
import dearpygui.dearpygui as dpg

from dwi_ml.gui.utils.argparse_to_gui import add_args_groups_to_gui
from dwi_ml.gui.utils.my_styles import STYLE_FIXED_WINDOW, \
    get_my_fonts_dictionary
from dwi_ml.gui.utils.window import callback_change_window
from dwi_ml.models.projects.learn2track_utils import get_all_args_groups_learn2track


def callback_ok_get_args_l2t(_, __, args):
    # user_data = args
    all_values = {}
    for arg_name in args.keys():
        name = params['dest'] if 'dest' in params else arg_name
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
        with dpg.window(**STYLE_FIXED_WINDOW,
                        tag='train_l2t_window',
                        no_open_over_existing_popup=False) as l2t_window:
            dpg.set_primary_window(l2t_window, True)

            # Top: Back button
            dpg.add_button(label='<-- Back', callback=callback_change_window, user_data=(l2t_window, main_window))

            # 1. Main title.
            title = dpg.add_text('\nLEARN2TRACK:\n\n')

            # 2. Getting all L2T args (same as in l2t_train_model).
            groups = get_all_args_groups_learn2track()
            for group, group_args in groups.items():
                add_args_groups_to_gui(group_args, group)

            # 3. Output script file + ok button.
            dpg.add_text('\n\n')
            dpg.add_button(
                label='Click here to choose where to save your file',
                indent=300, width=600,
                callback=lambda: dpg.show_item("file_dialog_output_script"))
            dpg.add_text("\n\n\n\n")
            dpg.add_button(label='Create my script!', indent=1000,
                           tag='create_l2t_train_script',
                           callback=callback_ok_get_args_l2t,
                           user_data=groups, height=50)

        # Bind all fonts. NOT WORKING??
        dpg.bind_item_font(title, my_fonts['main_title'])


def create_l2t_train_script(groups):
    script = "l2t_train_model.py "
    for group, group_args in groups.items():
        for arg_name, value in group_args.items():
            if value is not None:
                script += arg_name + ' ' + str(value)
            elif not arg_name[0:2] == '--':
                print("Some required values are not defined!")

    print(script)
