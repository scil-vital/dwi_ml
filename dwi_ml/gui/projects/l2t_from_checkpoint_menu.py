# -*- coding: utf-8 -*-
import dearpygui.dearpygui as dpg

from dwi_ml.gui.utils.my_styles import fixed_window_options


def open_l2t_from_checkpoint_window():
    with dpg.window(**fixed_window_options):
        dpg.add_text("NOT READY")
