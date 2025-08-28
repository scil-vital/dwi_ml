
# -*- coding: utf-8 -*-

# Top level
HELP_AND_INPUT_GROUP = '_help_and_input'

# Side by side:
# Tables because dpg.group does not respect width.
HELP_TABLE = '_help'
INPUTS_AND_OPTIONS_TABLE = '_inputs_and_checkbox'
INPUTS_CELL = '_inputs'
OPTIONS_CELL = '_options'

# In the inputs group: argname +
# FILE_DIALOG_BUTTON  or
OPTION_DEFAULT_TEXTBOX = '_notSelectedTxt'
# or 'narg0', 'narg1', etc.

# In the options group, tag = argname +
OPTION_CHECKBOX = '_default_checkbox'
NARGS_GROUP = '_nargs'
ADD_NARGS_BUTTON = '_add_narg_button'
REMOVE_NARGS_BUTTON = '_remove_narg_button'

# Text replacing value for action='store_true', 'store_false'
TEXT_SELECTED = "Selected!"
TEXT_UNSELECTED = "(not selected)"