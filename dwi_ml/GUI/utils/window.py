import dearpygui.dearpygui as dpg

from dwi_ml.GUI.styles.my_styles import get_global_theme, WINDOW_WIDTH


def start_dpg():
    dpg.create_context()
    dpg.create_viewport(title='Welcome to dwi_ml', width=WINDOW_WIDTH,
                        height=800)
    global_theme = get_global_theme()
    dpg.bind_theme(global_theme)
    dpg.setup_dearpygui()


def show_and_end_dpg():
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


def callback_change_window(_, __, user_data):
    current_window, next_window = user_data
    dpg.hide_item(current_window)
    dpg.show_item(next_window)
    dpg.set_primary_window(next_window, True)
