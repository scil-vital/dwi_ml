#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def test_help_option(script_runner):
    ret = script_runner.run('dwiml_divide_volume_into_blocs.py',
                            '--help')
    assert ret.success
