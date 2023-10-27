#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def test_help_option(script_runner):
    ret = script_runner.run('dwiml_compute_connectivity_matrix_from_blocs.py', '--help')
    assert ret.success
