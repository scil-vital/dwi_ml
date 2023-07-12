import dearpygui.dearpygui as dpg

from dwi_ml.gui.utils.my_styles import get_global_theme


def start_dpg():
    dpg.create_context()
    dpg.create_viewport(title='Welcome to dwi_ml', width=1300, height=800)
    global_theme = get_global_theme()
    dpg.bind_theme(global_theme)
    dpg.setup_dearpygui()


def show_and_end_dpg():
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


def toggle_full_screen():
    # Not working: toggles fullscreen for the main viewport, but not for the
    # window inside.
    dpg.add_menu_item(label="Toggle Fullscreen",
                      callback=lambda: dpg.toggle_viewport_fullscreen())


def callback_change_window(_, __, user_data):
    current_window, next_window = user_data
    dpg.hide_item(current_window)
    dpg.show_item(next_window)
    dpg.set_primary_window(next_window, True)
