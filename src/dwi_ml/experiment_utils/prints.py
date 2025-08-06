# -*- coding: utf-8 -*-
from collections.abc import Mapping


def format_dict_to_str(d, indent=1, keys_only=False):
    indentation = indent * "    "
    return ("\n" + indentation) + ("\n" + indentation).join(
        "{!r}: {},".format(k, _format_val_to_str(v, indent+1, keys_only))
        for k, v in d.items())


def _format_val_to_str(v, indent, keys_only=False):
    if isinstance(v, Mapping):
        return format_dict_to_str(v, indent, keys_only)
    elif keys_only:
        return ''
    else:
        return v
