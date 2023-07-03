# -*- coding: utf-8 -*-
import dearpygui.dearpygui as dpg

from dwi_ml.gui.utils.file_dialogs import params_file_dialogs, \
    callback_file_dialog_single_file
from dwi_ml.gui.utils.my_styles import fixed_window_options, \
    get_my_fonts_dictionary
from dwi_ml.gui.utils.window import callback_change_window


def open_menu_create_hdf5():
    main_window = dpg.get_active_window()
    dpg.hide_item(main_window)

    if dpg.does_item_exist('create_hdf5_window'):
        dpg.set_primary_window('create_hdf5_window', True)
        dpg.show_item('create_hdf5_window')
    else:
        # Create the HDF5 window.
        my_fonts = get_my_fonts_dictionary()
        with dpg.window(**fixed_window_options,
                        tag='create_hdf5_window') as hdf5_window:
            dpg.set_primary_window(hdf5_window, True)

            dpg.add_button(label='<-- Back', callback=callback_change_window,
                           user_data=(hdf5_window, main_window))

            title = dpg.add_text('\nCREATE YOUR HDF5:\n\n')
            dpg.bind_font(my_fonts['default'])
            dpg.bind_item_font(title, my_fonts['main_title'])  # NOT WORKING?

            dpg.add_button(label="Choose saving path", indent=50,
                           callback=open_file_dialog_hdf5_create_dir)


def open_file_dialog_hdf5_create_dir():
    file_dialog_name = "file_dialog_hdf5_dir"
    if dpg.does_item_exist(file_dialog_name):
        dpg.show_item(file_dialog_name)
    else:
        dpg.add_file_dialog(
            directory_selector=False, tag=file_dialog_name,
            **params_file_dialogs, callback=callback_file_dialog_single_file,
            user_data=UNDEFINED_YET)
