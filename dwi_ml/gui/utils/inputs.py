# -*- coding: utf-8 -*-

def assert_single_choice_file_dialog(app_data):
    chosen_paths = list(app_data['selections'].values())
    if len(chosen_paths) > 1:
        raise NotImplementedError("BUG IN OUR CODE? SHOULD NOT ALLOW "
                                  "MULTIPLE SELECTION")
    if len(chosen_paths) == 0:
        # toDo. Add as pop-up.
        print("      PLEASE CHOOSE A DIRECTORY.")
        return
