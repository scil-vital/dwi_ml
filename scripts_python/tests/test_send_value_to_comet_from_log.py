#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def test_help_option(script_runner):
    ret = script_runner.run('dwiml_send_value_to_comet_from_log.py', '--help')
    assert ret.success


def test_execution(script_runner):
    # Impossible
    pass
