# -*- coding: utf-8 -*-
import logging


def format_arg(argname, help, metavar=None, type=str, default=None,
               action=None, const=None, nargs=None, dest=None, choices=None):

    if action is not None and action != 'store_true':
        raise NotImplementedError("NOT READY")

    # Format metavar
    optional = True if argname[0] == '-' else False
    if (metavar is None and optional and
            action not in ['store_true', 'store_false']):
        metavar = argname.replace('-', '')

    # Modifying metavar if 'nargs' is set.
    if metavar is not None and nargs is not None:
        if isinstance(nargs, int):
            metavar = (metavar + ' ') * nargs
        elif nargs == '?':  # i.e. either default or one value. Uses const.
            metavar = '[' + metavar + ']'
        elif nargs == '*':  # Between 0 and inf nb.
            metavar = '[' + metavar + '... ]'
        elif nargs == '+':  # Between 1 and inf nb.
            metavar = metavar + ' [' + metavar + '... ]'

    return {'help': help, 'metavar': metavar, 'type': type, 'default': default,
            'const': const, 'nargs': nargs, 'dest': dest, 'choices': choices,
            'action': action}


class ArgparserForGui:
    def __init__(self, description=None):
        logging.debug("    GUI.args_management.argparse_equivalent: "
                      "Initializing ArgparserForGui")
        self.is_subgroup = False
        self.description = description
        self.main_required_args_dict = {}  # type: dict[str, dict]
        self.main_optional_args_dict = {}  # type: dict[str, dict]
        self.exclusive_groups = []  # type: List[MutuallyExclusiveGroup]
        self.groups_of_args = []  # type: List[ArgparserForGui]

    def add_argument(self, argname, **kwargs):
        optional = True if argname[0] == '-' else False
        if optional:
            self.main_optional_args_dict[argname] = format_arg(argname, **kwargs)
        else:
            if self.is_subgroup:
                raise NotImplementedError("I don't know how to manage "
                                          "required args from inside a group!"
                                          "Where do they go as positional args?")
            self.main_required_args_dict[argname] = format_arg(argname, **kwargs)

    def add_argument_group(self, desc):
        g = ArgparserForGui(desc)
        g.is_subgroup = True
        self.groups_of_args.append(g)
        return g

    def add_mutually_exclusive_group(self, required=False):
        g = MutuallyExclusiveGroup(required)
        self.exclusive_groups.append(g)
        return g

    def get_all_optional_argnames(self):
        names = list(self.main_optional_args_dict.keys())
        for eg in self.exclusive_groups:
            names.extend(list(eg.arguments_list.keys()))
        for eg in self.groups_of_args:
            names.extend(eg.get_all_optional_argnames())
        return names

    def get_all_required_names(self):
        names = list(self.main_required_args_dict.keys())
        # Currently, this should be empty:
        for eg in self.groups_of_args:
            names.extend(list(eg.main_required_args_dict.keys()))
        return names


class MutuallyExclusiveGroup:
    def __init__(self, required):
        self.required = required
        self.arguments_list = {}  # type: dict[str, dict]

    def add_argument(self, argname, **kwargs):
        self.arguments_list[argname] = format_arg(argname, **kwargs)
