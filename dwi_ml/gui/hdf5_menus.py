# -*- coding: utf-8 -*-
import dearpygui.dearpygui as dpg

from dwi_ml.data.hdf5.utils import get_hdf5_args_groups
from dwi_ml.gui.utils.argparse_to_gui import add_args_groups_to_gui
from dwi_ml.gui.utils.file_dialogs import \
    add_file_dialog_input_group
from dwi_ml.gui.utils.gui_popup_message import show_infobox
from dwi_ml.gui.utils.gui_to_argparse import get_all_values
from dwi_ml.gui.utils.my_styles import STYLE_FIXED_WINDOW, \
    get_my_fonts_dictionary
from dwi_ml.gui.utils.window import callback_change_window


TAG_HDF_SCRIPT_PATH = 'hdf5_creation_script'


def open_menu_create_hdf5():
    main_window = dpg.get_active_window()
    dpg.hide_item(main_window)

    if dpg.does_item_exist('create_hdf5_window'):
        dpg.set_primary_window('create_hdf5_window', True)
        dpg.show_item('create_hdf5_window')
    else:
        # Create the HDF5 window.
        my_fonts = get_my_fonts_dictionary()
        with dpg.window(**STYLE_FIXED_WINDOW,
                        tag='create_hdf5_window') as hdf5_window:
            dpg.set_primary_window(hdf5_window, True)

            dpg.add_button(label='<-- Back', callback=callback_change_window,
                           user_data=(hdf5_window, main_window))

            title = dpg.add_text('\nCREATE YOUR HDF5:\n\n')
            dpg.bind_font(my_fonts['default'])
            dpg.bind_item_font(title, my_fonts['main_title'])  # NOT WORKING?

            # Creating known file dialogs.
            #  If arg name in the script changes and we don't make the changes
            #  here, the file dialog will not be triggered and the arg will
            #  simply show as a str input.

            # All these are required in our script:
            # - dwi_ml_ready_folder
            # - out_hdf5_file => .hdf5
            # - config_file => .json
            # - subjects lists => .txt
            known_file_dialogs = {
                'dwi_ml_ready_folder': ['unique_folder', ],
                'out_hdf5_file': ['unique_file', ['.hdf5']],
                'config_file': ['unique_file', ['.json']],
                'training_subjs': ['unique_file', ['.txt']],
                'validation_subjs': ['unique_file', ['.txt']],
                'testing_subjs': ['unique_file', ['.txt']]}

            # Ok! Adding all buttons
            groups = get_hdf5_args_groups()
            add_args_groups_to_gui(groups, known_file_dialogs)

            dpg.add_text("\n\n\n\n")
            with dpg.group(horizontal=True):
                dpg.add_text('Choose where to save output script: ')
                add_file_dialog_input_group(
                    TAG_HDF_SCRIPT_PATH, ['unique_file', ['.sh']])

            dpg.add_text("\n")
            dpg.add_button(
                label='OK! Create my script!', indent=1005, width=200,
                height=100, callback=_create_hdf5_creation_script,
                user_data=groups)


def _create_hdf5_creation_script(_, __, user_data):
    # 1. Verify that user defined where to save the script.
    out_path = dpg.get_value(TAG_HDF_SCRIPT_PATH)
    if out_path is None:
        show_infobox("Missing value", "Please choose an output file for the "
                                      "created bash script.")

    # 2. Get values. Verify that no warning message popped-up (if so, returns
    # None)
    all_values = get_all_values(user_data)
    if all_values is None:
        return

    DEBUG = True
    if DEBUG:
        print("WILL WRITE:")
        print('dwiml_create_hdf5_dataset.py \n \\' + '\n'.join(all_values))
        print("IN : ", out_path)
        return

    # toDo
    script_out_file = 'TO BE DETERMINED'
    with open(script_out_file, 'w') as f:
        f.write('dwiml_create_hdf5_dataset.py \n')
        f.writelines(all_values)
