# -*- coding: utf-8 -*-
import os

import dearpygui.dearpygui as dpg

from dwi_ml.gui.sub_menus_train_model import open_l2t_from_checkpoint_window, \
    open_tto_from_checkpoint_window, open_ttst_from_checkpoint_window, \
    open_l2t_window, open_tto_window, open_ttst_window
from dwi_ml.gui.utils.my_styles import get_my_fonts_dictionary
from dwi_ml.gui.utils.window import start_dpg, show_and_end_dpg
from dwi_ml.io_utils import verify_checkpoint_exists, verify_which_model_in_path
from dwi_ml.models.projects.learn2track_model import Learn2TrackModel
from dwi_ml.models.projects.transformer_models import OriginalTransformerModel, \
    TransformerSrcAndTgtModel


def prepare_main_menu():
    start_dpg()

    # ---------
    # Preparing file dialgogs. There seems to be a bug with dearpygui, I cannot
    # separate this from main code (i.e. encapsulate).
    # See my issue. https://github.com/hoffstadt/DearPyGui/issues/2120
    # ---------
    def checkpoint_file_dialog_ok_callback(_, app_data):
        subcallback_checkpoint_file_dialog_ok(app_data)

    def checkpoint_file_dialog_cancel_callback(_, app_data):
        pass

    home = os.path.expanduser('~')
    file_dialog = dpg.add_file_dialog(
        directory_selector=True, show=False, default_path=home,
        callback=checkpoint_file_dialog_ok_callback,
        tag="file_dialog_resume_from_checkpoint", file_count=1,
        cancel_callback=checkpoint_file_dialog_cancel_callback,
        width=700, height=400)

    # Trying to set a different color. Not working...
    with dpg.theme() as item_theme:
        with dpg.theme_component(0):
            gray = (78, 78, 78, 255)
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, gray)
    dpg.bind_item_theme(file_dialog, item_theme)

    # ---------
    # Ok. Now preparing the window
    # ---------
    with dpg.window(tag="Primary Window"):
        title0 = dpg.add_text("                                               "
                              "                 WELCOME TO DWI_ML")

        ##########
        # 1. Training
        ##########
        title1 = dpg.add_text("\n\nTrain a new model:")
        dpg.add_button(label="Learn2track", indent=50,
                       callback=callback_train_l2t)
        dpg.add_button(label="Transforming tractography: original model",
                       indent=50, callback=callback_train_tto)
        dpg.add_button(label="Transforming tractography: source-target model",
                       indent=50, callback=callback_train_ttst)

        ##########
        # 2. Resume from checkpoint.
        ##########
        title2 = dpg.add_text("\n\nContinue training model from an existing "
                              "experiment (resume from checkpoint).")
        dpg.add_button(label="Select your experiment's directory",
                       callback=lambda: dpg.show_item("file_dialog_resume_from_checkpoint"),
                       indent=50)

        ##########
        # 3. Visualize logs
        ##########
        title3 = dpg.add_text("\n\nVisualization")

        ##########
        # 4. Track from a model.
        ##########
        title4 = dpg.add_text("\n\nTrack from a model")

        my_fonts = get_my_fonts_dictionary()
        dpg.bind_font(my_fonts['default'])
        dpg.bind_item_font(title0, my_fonts['main_title'])
        for title in [title1, title2, title3, title4]:
            dpg.bind_item_font(title, my_fonts['title'])

    show_and_end_dpg()


def callback_train_l2t():
    print("You want to train a new Learn2track model")
    open_l2t_window()


def callback_train_tto():
    open_tto_window()


def callback_train_ttst():
    open_ttst_window()


def subcallback_checkpoint_file_dialog_ok(app_data):
    # Verifying for multiple selections, just in case.
    # But not using: bug in dearpygui: the selected folder is added twice
    # at the end of the path.
    chosen_paths = list(app_data['selections'].values())
    if len(chosen_paths) > 1:
        raise NotImplementedError("BUG IN OUR CODE? SHOULD NOT ALLOW "
                                  "MULTIPLE SELECTION")
    if len(chosen_paths) == 0:
        print("      PLEASE CHOOSE A DIRECTORY.")
        return

    chosen_path = app_data['current_path']
    # toDo: if raises a FileNotFoundError, show a pop-up warning?
    #  Currently prints the warning in terminal.
    checkpoint_path = verify_checkpoint_exists(chosen_path)

    model_dir = os.path.join(checkpoint_path, 'model')
    model_type, _ = verify_which_model_in_path(model_dir)
    print("Model's class: {}".format(model_type))

    if model_type == Learn2TrackModel.__name__:
        open_l2t_from_checkpoint_window()
    elif model_type == OriginalTransformerModel.__name__:
        open_tto_from_checkpoint_window()
    elif model_type == TransformerSrcAndTgtModel.__name__:
        open_ttst_from_checkpoint_window()
    else:
        raise ValueError("This type of model is not managed by our DWIML' GUI.")
