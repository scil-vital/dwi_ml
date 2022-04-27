#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def test_help_option(script_runner):
    ret = script_runner.run('dwiml_visualize_logs.py', '--help')
    assert ret.success
