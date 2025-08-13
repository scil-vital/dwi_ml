# -*- coding: utf-8 -*-
import logging

store_options = ['store_true', 'store_false']


def format_arg(argname, help, metavar=None, type=str, default=None,
               action=None, const=None, nargs=None, dest=None, choices=None):

    if action is not None and action != 'store_true':
        raise NotImplementedError("NOT READY")

    # Format metavar
    optional = True if argname[0] == '-' else False
    if metavar is None and optional and action not in store_options:
        # .upper() to put in capital like real argparser, but I don't like that
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
        self.description = description
        self.required_args_dict = {}
        self.optional_args_dict = {}
        self.exclusive_groups = []  # type: List[MutuallyExclusiveGroup]

        # Should never be changed if self is created with add_argument_group.
        # But we rely on the fact that the scripts are used through the real
        # argparser. No need to test.
        self.groups_of_args = []  # type: List[ArgparserForGui]

    def add_argument(self, argname, **kwargs):
        optional = True if argname[0] == '-' else False
        if optional:
            self.optional_args_dict[argname] = format_arg(argname, **kwargs)
        else:
            self.required_args_dict[argname] = format_arg(argname, **kwargs)

    def add_argument_group(self, desc):
        g = ArgparserForGui(desc)
        self.groups_of_args.append(g)
        return g

    def add_mutually_exclusive_group(self, required=False):
        g = MutuallyExclusiveGroup(required)
        self.exclusive_groups.append(g)
        return g


class MutuallyExclusiveGroup:
    def __init__(self, required):
        self.required = required
        self.arguments_list = {}

    def add_argument(self, argname, **kwargs):
        self.arguments_list[argname] = format_arg(argname, **kwargs)
