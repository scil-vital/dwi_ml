# -*- coding: utf-8 -*-

import dearpygui.dearpygui as dpg


def create_DPG_tools_dropdown_menu():
    """
    A list of DPG tools that can be useful to debug and improve GUI. Will add
    a "DPG Tools" section in your menu, with documentation and other DPG
    utilities. Should be added to a menu bar (i.e. used inside a
        with dpg.menu_bar():
    environment).
    """
    with dpg.menu(label="DPG Tools"):
        dpg.add_menu_item(label="Show About DPG",
                          callback=lambda: dpg.show_tool(dpg.mvTool_About))
        dpg.add_menu_item(label="Show DPG Metrics",
                          callback=lambda: dpg.show_tool(dpg.mvTool_Metrics))
        dpg.add_menu_item(label="Show DPG Documentation",
                          callback=lambda: dpg.show_tool(dpg.mvTool_Doc))
        dpg.add_menu_item(label="Show DPG Debug",
                          callback=lambda: dpg.show_tool(dpg.mvTool_Debug))
        dpg.add_menu_item(label="Show DPG Item Registry",
                          callback=lambda: dpg.show_tool(dpg.mvTool_ItemRegistry))

