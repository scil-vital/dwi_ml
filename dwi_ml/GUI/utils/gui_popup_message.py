# -*- coding: utf-8 -*-

import dearpygui.dearpygui as dpg


def simply_close_infobox(sender, unused, user_data):
    if user_data[1]:
        clicked_ok = True
    else:
        clicked_cancel = True

    # delete window
    dpg.delete_item(user_data[0])


def show_infobox(title, message, ok_callback=simply_close_infobox,
                 add_cancel=False, cancel_callback=simply_close_infobox):
    # Code from here: https://github.com/hoffstadt/DearPyGui/discussions/1308

    # guarantee these commands happen in the same frame
    if add_cancel:
        indent = 325
    else:
        indent = 400

    with dpg.mutex():
        viewport_width = dpg.get_viewport_client_width()
        viewport_height = dpg.get_viewport_client_height()

        with dpg.window(label=title, modal=True, no_close=True,
                        no_open_over_existing_popup=False,
                        width=500) as modal_id:
            dpg.add_text(message, wrap=499)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Ok", width=75, indent=indent,
                               user_data=(modal_id, True),
                               callback=ok_callback)
                if add_cancel:
                    dpg.add_button(label="Cancel", width=75,
                                   user_data=(modal_id, False),
                                   callback=cancel_callback)

    # guarantee these commands happen in another frame
    dpg.split_frame()
    # But does not work when this is launched after my file dialog.
    width = dpg.get_item_width(modal_id)
    height = dpg.get_item_height(modal_id)
    dpg.set_item_pos(modal_id, [viewport_width // 2 - width // 2,
                                viewport_height // 2 - height // 2])
