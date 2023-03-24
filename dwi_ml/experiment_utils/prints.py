# -*- coding: utf-8 -*-
from collections.abc import Mapping


def format_dict_to_str(d, indent=1):
    indentation = indent * "    "
    return ("\n" + indentation) + ("\n" + indentation).join(
        "{!r}: {},".format(k, _format_val_to_str(v, indent+1))
        for k, v in d.items())


def _format_val_to_str(v, indent):
    if isinstance(v, Mapping):
        return format_dict_to_str(v, indent)
    else:
        return v
