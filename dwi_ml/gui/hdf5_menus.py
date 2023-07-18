# -*- coding: utf-8 -*-
import dearpygui.dearpygui as dpg

from dwi_ml.data.hdf5.utils import get_hdf5_args_groups
from dwi_ml.gui.utils.argparser_equivalent_for_gui import add_args_groups_to_gui, \
    style_input_item, get_all_values
from dwi_ml.gui.utils.file_dialogs import add_single_file_dialog, \
    add_single_folder_dialog
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

            # Creating required file dialogs
            # - dwi_ml_ready_folder
            # - out_hdf5_file => .hdf5
            # - config_file => .json
            # - subjects lists => .txt
            add_single_folder_dialog('dwi_ml_ready_folder')
            add_single_file_dialog('out_hdf5_file', extensions=['.hdf5'])
            add_single_file_dialog('config_file', extensions=['.json'])
            for subjs in ['training_subjs', 'validation_subjs', 'testing_subjs']:
                add_single_file_dialog(subjs, extensions=['.txt'])
            add_single_file_dialog('hdf5_creation_script', extensions=['.sh'])

            # Ok! Adding all buttons
            groups = get_hdf5_args_groups()
            add_args_groups_to_gui(
                groups, existing_file_dialogs=[
                    'dwi_ml_ready_folder', 'out_hdf5_file', 'config_file',
                    'training_subjs', 'validation_subjs', 'testing_subjs'])

            dpg.add_text("\n\n\n\n")
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label='Click here to select path', **style_input_item,
                    callback=lambda: dpg.show_item('hdf5_creation_script'))

                dpg.add_button(
                    label='OK! Create my script!', indent=1005, width=200,
                    height=100, callback=_create_hdf5_creation_script,
                    user_data=groups)


def _create_hdf5_creation_script(sender, app_data, user_data):
    all_values = get_all_values(user_data)

    if not all_values:
        # An error occured
        return

    # Create script
    script_out_file = dpg.get_value('hdf5_creation_script')

    print("WILL WRITE:")
    print('dwiml_create_hdf5_dataset.py ' + '\n'.join(all_values))

    print("IN : ", script_out_file)

    raise NotImplementedError
    with open(script_out_file, 'w') as f:
        f.write('dwiml_create_hdf5_dataset.py \n')
        f.writelines(all_values)
