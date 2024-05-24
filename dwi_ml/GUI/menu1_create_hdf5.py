# -*- coding: utf-8 -*-
import logging

import dearpygui.dearpygui as dpg

from dwi_ml.GUI.args_management.args_to_gui_all_args import add_args_to_gui
from dwi_ml.GUI.utils.file_dialogs import add_file_dialog_input_group
from dwi_ml.GUI.utils.gui_popup_message import show_infobox
from dwi_ml.GUI.styles.my_styles import STYLE_FIXED_WINDOW, \
    get_my_fonts_dictionary
from dwi_ml.GUI.styles.theme_inputs import add_color_legend
from dwi_ml.GUI.args_management.gui_to_argparse import get_all_values_as_str
from dwi_ml.GUI.utils.window import callback_change_window
from scripts_python.dwiml_create_hdf5_dataset import prepare_argparser

TAG_HDF_SCRIPT_PATH = 'hdf5_creation_script'


def open_menu_create_hdf5():
    logging.debug("GUI.menu1_create_hdf5.open_menu_create_hdf5(): "
                  "Preparing HDF5 window...")
    main_window = dpg.get_active_window()
    dpg.hide_item(main_window)

    if dpg.does_item_exist('create_hdf5_window'):
        # User had already reached this page but then left and came back.
        dpg.set_primary_window('create_hdf5_window', True)
        dpg.show_item('create_hdf5_window')
    else:
        # Create the HDF5 window.
        with dpg.window(**STYLE_FIXED_WINDOW,
                        tag='create_hdf5_window') as hdf5_window:
            dpg.set_primary_window(hdf5_window, True)

            dpg.add_button(label='<-- Back', callback=callback_change_window,
                           user_data=(hdf5_window, main_window))
            add_color_legend()

            main_title = dpg.add_text(
                "                                                            "
                "    CREATE A HDF5 DATASET\n\n")
            txt = dpg.add_text(
                "Choose your options. We will prepare a bash script for you "
                "that you can then run in command line.\n\n")
            dpg.add_text("Script description "
                         "(dwiml_create_hdf5_dataset.py --help):")

            # Main script description.
            gui_parser = prepare_argparser(from_gui=True)
            desc_text = dpg.add_text(gui_parser.description, indent=100)

            # Creating known file dialogs.
            #  If arg names in the script change and if we don't make the
            #  changes here, the file dialog will not be triggered and the arg
            #  will simply show as a str input.

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
            add_args_to_gui(gui_parser, known_file_dialogs)

            dpg.add_text("\n\n\n\n")
            with dpg.group(horizontal=True) as group:
                dpg.add_text('Choose where to save output command line '
                             '(bash .sh file): ')
                add_file_dialog_input_group(
                    TAG_HDF_SCRIPT_PATH, ['unique_file', ['.sh']],
                    input_parent=group, button_parent=group)

            dpg.add_text("\n")
            dpg.add_button(
                label='OK! Create my command line!', indent=1005, width=200,
                height=100, callback=__create_hdf5_script_callback,
                user_data=gui_parser)

        # Set Title fonts
        MY_FONTS = get_my_fonts_dictionary()
        dpg.bind_item_font(main_title, MY_FONTS['main_title'])
        dpg.bind_item_font(txt, MY_FONTS['section_title'])
        dpg.bind_item_font(desc_text, MY_FONTS['code'])


def __create_hdf5_script_callback(_, __, user_data):
    """
    user_data: the Argparser_for_GUI
    """
    logging.debug("GUI.menu1_create_hdf5.__create_hdf5_script_callback(): "
                  "Preparing script...")

    # 1. Verify that user defined where to save the script.
    out_path = dpg.get_value(TAG_HDF_SCRIPT_PATH)
    if out_path is None or out_path == '':
        show_infobox("Missing value", "Please choose an output file for the "
                                      "created bash script.")
        return
    elif out_path[-3:] != '.sh':
        show_infobox("Wrong filename", "Your output filename should be .sh, "
                                       "got '{}'".format(out_path))
        return

    logging.info("Will save the command line in {}.".format(out_path))

    # 2. Get values. Verify that no warning message popped-up (if so, returns
    # None)
    all_values = get_all_values_as_str(user_data)
    if all_values is None:
        # Showed a message box. Leaving now.
        return

    logging.debug("WILL WRITE:")
    logging.debug('dwiml_create_hdf5_dataset.py \n \\' + '\n'.join(all_values))
    logging.debug("IN : {}".format(out_path))

    raise NotImplementedError

    # toDo
    script_out_file = 'TO BE DETERMINED'
    with open(script_out_file, 'w') as f:
        f.write('dwiml_create_hdf5_dataset.py \n')
        f.writelines(all_values)
