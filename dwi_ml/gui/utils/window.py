import dearpygui.dearpygui as dpg
from dwi_ml.gui.utils.my_styles import set_global_theme


def start_dpg():
    dpg.create_context()
    dpg.create_viewport(title='Welcome to dwi_ml', width=1300, height=800)


def show_and_end_dpg():
    global_theme = set_global_theme()
    dpg.bind_theme(global_theme)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("Primary Window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()


def toggle_full_screen():
    # Not working: toggles fullscreen for the main viewport, but not for the
    # window inside.
    dpg.add_menu_item(label="Toggle Fullscreen",
                      callback=lambda: dpg.toggle_viewport_fullscreen())