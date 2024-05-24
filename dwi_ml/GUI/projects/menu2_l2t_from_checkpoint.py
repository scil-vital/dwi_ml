# -*- coding: utf-8 -*-
import dearpygui.dearpygui as dpg

from dwi_ml.GUI.styles.my_styles import STYLE_FIXED_WINDOW


def open_l2t_from_checkpoint_window():
    with dpg.window(**STYLE_FIXED_WINDOW):
        dpg.add_text("NOT READY")
