# -*- coding: utf-8 -*-

import dearpygui.dearpygui as dpg


def add_click_toggle_item():  # Adds a checkmark on the side when toggled.
    dpg.add_menu_item(label="Wait For Input", check=True,
                      callback=lambda s, a: dpg.configure_app(wait_for_input=a))


def create_settings_dropdown_menu():
    with dpg.menu(label="Settings"):
        dpg.add_menu_item(label="Show DPG Style Editor",
                          callback=lambda: dpg.show_tool(dpg.mvTool_Style))
        dpg.add_menu_item(label="Show DPG Font Manager",
                          callback=lambda: dpg.show_tool(dpg.mvTool_Font))
