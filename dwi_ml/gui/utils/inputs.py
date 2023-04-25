# -*- coding: utf-8 -*-

import dearpygui.dearpygui as dpg


def add_slider():
    dpg.add_slider_float(label="Slider Float")


def add_plus_or_minus_int_box():
    # Add an input box expecting an integer value. You can enter the value
    # yourself or click on +/- sign on the side.
    dpg.add_input_int(label="Input Int")


def add_dropdown_list_of_choice():
    dpg.add_combo(("Yes", "No", "Maybe"), label="Combo")


def add_scrolldown_text():
    with dpg.child_window(height=60, autosize_x=True,
                          delay_search=True):
        for i in range(10):
            dpg.add_text(f"Scrolling Text{i}")