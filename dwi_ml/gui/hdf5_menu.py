# -*- coding: utf-8 -*-
import dearpygui.dearpygui as dpg


def callback_ok_get_argshdf5(_, __, args):
    # user_data = args
    all_values = {}
    for arg_name in args.keys():
        all_values[arg_name] = dpg.get_value(arg_name)

    print("ALL VALUES: ")
    print(all_values)
    create_hdf5_creation_script(all_values)
