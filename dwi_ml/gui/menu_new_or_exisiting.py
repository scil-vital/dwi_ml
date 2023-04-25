# -*- coding: utf-8 -*-

import dearpygui.dearpygui as dpg

from dwi_ml.gui.utils.my_styles import get_my_fonts_dictionary


def launch_main_menu():
    def callback(sender, app_data):
        print('OK was clicked.')
        print("Sender: ", sender)
        print("App Data: ", app_data)

    def cancel_callback(sender, app_data):
        print('Cancel was clicked.')
        print("Sender: ", sender)
        print("App Data: ", app_data)

    dpg.add_file_dialog(
        directory_selector=True, show=False, callback=callback,
        tag="file_dialog_id",
        cancel_callback=cancel_callback, width=700, height=400)

    with dpg.window(tag="Primary Window"):
        title0 = dpg.add_text("                                               "
                              "                 WELCOME TO DWI_ML")

        ##########
        # 1. Training
        ##########
        title1 = dpg.add_text("\n\nTrain a new model:")
        dpg.add_button(label="Learn2track", indent=50)
        dpg.add_button(label="Transforming tractography: original model",
                       indent=50)
        dpg.add_button(label="Transforming tractography: source-target model",
                       indent=50)

        ##########
        # 2. Resume from checkpoint.
        ##########
        title2 = dpg.add_text("\n\nContinue training model from an existing "
                              "experiment (resume from checkpoint).")
        dpg.add_button(label="Select your experiment's directory",
                       callback=lambda: dpg.show_item("file_dialog_id"),
                       indent=50)

        ##########
        # 3. Visualize logs
        ##########
        title3 = dpg.add_text("\n\nVisualization")

        ##########
        # 4. Track from a model.
        ##########
        title4 = dpg.add_text("\n\nTrack from a model")
        dpg.add_button(label="Select your experiment's directory. "
                             "(We will load its 'best model')",
                       callback=lambda: dpg.show_item("file_dialog_id"),
                       indent=50)

        my_fonts = get_my_fonts_dictionary()
        dpg.bind_font(my_fonts['default'])
        dpg.bind_item_font(title0, my_fonts['main_title'])
        for title in [title1, title2, title3, title4]:
            dpg.bind_item_font(title, my_fonts['title'])
