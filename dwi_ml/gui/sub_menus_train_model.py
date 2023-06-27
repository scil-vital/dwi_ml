# -*- coding: utf-8 -*-

import dearpygui.dearpygui as dpg

from dwi_ml.gui.utils.my_styles import fixed_window_options


def open_l2t_from_checkpoint_window():
    with dpg.window(**fixed_window_options):
        pass


def open_tto_from_checkpoint_window():
    print("Allo TTO")


def open_ttst_from_checkpoint_window():
    print("Allo TTST")


def open_l2t_window():
    with dpg.popup(dpg.last_item(), mousebutton=dpg.mvMouseButton_Left,
                   modal=True, tag="modal_id", width=1000, height=800):
        print("Creating popup")
        dpg.add_button(label="Close", callback=lambda: dpg.configure_item("modal_id", show=False))


def open_tto_window():
    print("aallo tto")


def open_ttst_window():
    print("aallo ttst")
