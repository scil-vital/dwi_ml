# -*- coding: utf-8 -*-
import os
from pathlib import Path

import dearpygui.dearpygui as dpg

STYLE_FIXED_WINDOW = {'no_move': True,
                      'no_collapse': True,
                      'no_close': True,
                      'no_title_bar': True,
                      'pos': [0, 0]
                      }

WINDOW_WIDTH = 1300

INDENT_ARGPARSE_NAME = 40
INDENT_ITEM = 550
WIDTH_ITEM = 400

INDENT_SET_IT_OPTION = INDENT_ITEM + WIDTH_ITEM + 10
INDENT_MORE_NARG = INDENT_SET_IT_OPTION + 100

STYLE_ARGPARSE_HELP = {
    'indent': 100,
    'color': (151, 151, 151, 255)
}
NB_DOTS = 115

# Defining a few colors. For help, use dpg.show_style_editor()
# 4th value = alpha
white = (255, 255, 255, 255)
light_gray = (151, 151, 151, 255)
gray = (78, 78, 78, 255)
dark_gray = (50, 50, 50, 255)
black = (25, 25, 25, 255)
transparent = (0, 0, 0, 0)
blue_hover = (12, 203, 235, 255)
blue_background = (80, 150, 180, 255)
chosen_purple = (130, 75, 177, 255)
pink_for_tests = (210, 8, 252, 255)
required_red = (242, 8, 8, 255)

# Defining values that will be used as global constants to avoid re-defining
# many times the same theme for each item.
global_modified_theme = None
global_non_modified_theme = None
global_required_theme = None
global_fonts = None


def add_color_legend():
    dpg.add_text("Values in red are required",
                 color=required_red, indent=WINDOW_WIDTH - 300)
    dpg.add_text("Values in purple already have been set.",
                 color=chosen_purple, indent=WINDOW_WIDTH - 300)


def callback_log_value_change_style(sender, app_data, _):
    """
    Callback intended for input items belonging to a mutually exclusive group.
    By default, they are always optional and thus are associated to a default
    checkbox.
    """
    if str(dpg.get_item_type(sender)) == 'mvAppItemType::mvCheckbox':
        # Checkbox.
        raise NotImplementedError("We have not prepared this callback for "
                                  "checkbox values.")
    else:
        if app_data is None or app_data == '':
            # User has deleted the value. Required again.
            dpg.bind_item_theme(sender, get_required_theme())
        else:
            # User has entered some value
            dpg.bind_item_theme(sender, get_modified_theme())


def _create_item_theme(color):
    link_theme = dpg.add_theme()
    with dpg.theme_component(0, parent=link_theme):
        dpg.add_theme_color(dpg.mvThemeCol_FrameBg, color)
        dpg.add_theme_color(dpg.mvThemeCol_Button, color)
    return link_theme


def get_modified_theme() -> int:
    global global_modified_theme
    if global_modified_theme is None:
        global_modified_theme = _create_item_theme(chosen_purple)
    return global_modified_theme


def get_non_modified_theme() -> int:
    global global_non_modified_theme
    if global_non_modified_theme is None:
        global_non_modified_theme = _create_item_theme(gray)
    return global_non_modified_theme


def get_required_theme() -> int:
    global global_required_theme
    if global_required_theme is None:
        global_required_theme = _create_item_theme(required_red)
    return global_required_theme


def get_my_fonts_dictionary():
    global global_fonts
    if global_fonts is not None:
        return global_fonts
    else:
        current_path = __file__
        gui_utils_path = Path(current_path).parent.absolute()
        font_path = os.path.join(gui_utils_path, "fonts/NotoSerifCJKjp-Medium.otf")

        with dpg.font_registry():
            # first argument is the path to the .ttf or .otf file
            default_font = dpg.add_font(font_path, 18)
            group_font = dpg.add_font(font_path, 22)
            title_font = dpg.add_font(font_path, 30)
            main_title_font = dpg.add_font(font_path, 40)

        global_fonts = {'default': default_font,
                        'group_title': group_font,
                        'section_title': title_font,
                        'main_title': main_title_font}
        return global_fonts


def get_global_theme():
    # Copied from :
    # https://github.com/hoffstadt/DearPyGui_Ext/blob/master/dearpygui_ext/themes.py
    # dpg.show_style_editor()

    with dpg.theme() as global_theme:
        with dpg.theme_component(0):
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 5, 5)

            dpg.add_theme_color(dpg.mvThemeCol_Text, white)
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, light_gray)

            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, dark_gray)
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, dark_gray)
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, black)

            dpg.add_theme_color(dpg.mvThemeCol_Border, gray)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, gray)

            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, light_gray)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, blue_hover)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, chosen_purple)

            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, black)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, chosen_purple)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgCollapsed, black)

            dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg, dark_gray)

            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, dark_gray)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, gray)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, light_gray)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive, light_gray)

            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, chosen_purple)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, blue_hover)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, blue_hover)
            dpg.add_theme_color(dpg.mvThemeCol_Button, blue_background)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, blue_hover)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, chosen_purple)

            dpg.add_theme_color(dpg.mvThemeCol_Header,
                                (0.26 * 255, 0.59 * 255, 0.98 * 255, 0.31 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, blue_hover)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, chosen_purple)

            dpg.add_theme_color(dpg.mvThemeCol_Separator, dark_gray)
            dpg.add_theme_color(dpg.mvThemeCol_SeparatorHovered, gray)
            dpg.add_theme_color(dpg.mvThemeCol_SeparatorActive, gray)

            dpg.add_theme_color(dpg.mvThemeCol_ResizeGrip, black)
            dpg.add_theme_color(dpg.mvThemeCol_ResizeGripHovered, dark_gray)
            dpg.add_theme_color(dpg.mvThemeCol_ResizeGripActive, dark_gray)

            dpg.add_theme_color(dpg.mvThemeCol_Tab, dark_gray)
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, blue_hover)
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, chosen_purple)
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocused, dark_gray)
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocusedActive, chosen_purple)

            dpg.add_theme_color(dpg.mvThemeCol_DockingPreview, blue_background)
            dpg.add_theme_color(dpg.mvThemeCol_DockingEmptyBg, dark_gray)

            dpg.add_theme_color(dpg.mvThemeCol_PlotLines, blue_background)
            dpg.add_theme_color(dpg.mvThemeCol_PlotLinesHovered, blue_hover)
            dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, gray)
            dpg.add_theme_color(dpg.mvThemeCol_PlotHistogramHovered, blue_hover)
            dpg.add_theme_color(dpg.mvThemeCol_TableHeaderBg, dark_gray)
            dpg.add_theme_color(dpg.mvThemeCol_TableBorderStrong, light_gray)
            dpg.add_theme_color(dpg.mvThemeCol_TableBorderLight, gray)
            dpg.add_theme_color(dpg.mvThemeCol_TableRowBg, transparent)
            dpg.add_theme_color(dpg.mvThemeCol_TableRowBgAlt, transparent)
            dpg.add_theme_color(dpg.mvThemeCol_TextSelectedBg,
                                (0.26 * 255, 0.59 * 255, 0.98 * 255, 0.35 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_DragDropTarget,
                                (1.00 * 255, 1.00 * 255, 0.00 * 255, 0.90 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_NavHighlight,
                                (0.26 * 255, 0.59 * 255, 0.98 * 255, 1.00 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_NavWindowingHighlight,
                                (1.00 * 255, 1.00 * 255, 1.00 * 255, 0.70 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_NavWindowingDimBg,
                                (0.80 * 255, 0.80 * 255, 0.80 * 255, 0.20 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_ModalWindowDimBg,
                                (0.80 * 255, 0.80 * 255, 0.80 * 255, 0.35 * 255))
            dpg.add_theme_color(dpg.mvPlotCol_FrameBg,
                                (1.00 * 255, 1.00 * 255, 1.00 * 255, 0.07 * 255),
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_PlotBg,
                                (0.00 * 255, 0.00 * 255, 0.00 * 255, 0.50 * 255),
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_PlotBorder,
                                (0.43 * 255, 0.43 * 255, 0.50 * 255, 0.50 * 255),
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_LegendBg,
                                (0.08 * 255, 0.08 * 255, 0.08 * 255, 0.94 * 255),
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_LegendBorder,
                                (0.43 * 255, 0.43 * 255, 0.50 * 255, 0.50 * 255),
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_LegendText,
                                (1.00 * 255, 1.00 * 255, 1.00 * 255, 1.00 * 255),
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_TitleText,
                                (1.00 * 255, 1.00 * 255, 1.00 * 255, 1.00 * 255),
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_InlayText,
                                (1.00 * 255, 1.00 * 255, 1.00 * 255, 1.00 * 255),
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_XAxis,
                                (1.00 * 255, 1.00 * 255, 1.00 * 255, 1.00 * 255),
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_XAxisGrid,
                                (1.00 * 255, 1.00 * 255, 1.00 * 255, 0.25 * 255),
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_YAxis,
                                (1.00 * 255, 1.00 * 255, 1.00 * 255, 1.00 * 255),
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_YAxisGrid,
                                (1.00 * 255, 1.00 * 255, 1.00 * 255, 0.25 * 255),
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_YAxis2,
                                (1.00 * 255, 1.00 * 255, 1.00 * 255, 1.00 * 255),
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_YAxisGrid2,
                                (1.00 * 255, 1.00 * 255, 1.00 * 255, 0.25 * 255),
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_YAxis3,
                                (1.00 * 255, 1.00 * 255, 1.00 * 255, 1.00 * 255),
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_YAxisGrid3,
                                (1.00 * 255, 1.00 * 255, 1.00 * 255, 0.25 * 255),
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_Selection,
                                (1.00 * 255, 0.60 * 255, 0.00 * 255, 1.00 * 255),
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_Query,
                                (0.00 * 255, 1.00 * 255, 0.44 * 255, 1.00 * 255),
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_Crosshairs,
                                (1.00 * 255, 1.00 * 255, 1.00 * 255, 0.50 * 255),
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, (50, 50, 50, 255),
                                category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered,
                                (75, 75, 75, 255), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected,
                                (75, 75, 75, 255), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_NodeOutline, (100, 100, 100, 255),
                                category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_TitleBar, (41, 74, 122, 255),
                                category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered, (66, 150, 250, 255),
                                category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected,
                                (66, 150, 250, 255), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_Link, (61, 133, 224, 200),
                                category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_LinkHovered, (66, 150, 250, 255),
                                category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_LinkSelected, (66, 150, 250, 255),
                                category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_Pin, (53, 150, 250, 180),
                                category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_PinHovered, (53, 150, 250, 255),
                                category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_BoxSelector, (61, 133, 224, 30),
                                category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_BoxSelectorOutline,
                                (61, 133, 224, 150), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_GridBackground, (40, 40, 50, 200),
                                category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_GridLine, (200, 200, 200, 40),
                                category=dpg.mvThemeCat_Nodes)

    # toDo: Activate FrameBorder
    #  Set button colors

    return global_theme
