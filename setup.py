#!/usr/bin/env python
"""Installation script for the DWI_ML package."""

from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path
from os.path import join as pjoin
import glob

from setup_helpers import read_vars_from


# Read package information
info = read_vars_from(pjoin('dwi_ml', 'info.py'))


this_directory = path.abspath(path.dirname(__file__))

dwi_ml_readme_path = pjoin(this_directory, 'README.rst')
with open(dwi_ml_readme_path, encoding='utf-8') as f:
    long_description = f.read()

setup(
    name=info.NAME,
    version=info.VERSION,
    description=info.DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url=info.URL,
    project_urls={'Bug tracker': info.BUG_TRACKER,
                  'Documentation': info.DOCUMENTATION,
                  'Source code': info.SOURCE_CODE},
    license=info.LICENSE,
    author=info.AUTHOR,
    author_email=info.AUTHOR_EMAIL,
    classifiers=info.CLASSIFIERS,
    keywords=info.KEYWORDS,
    maintainer=info.MAINTAINER,
    provides=info.PROVIDES,
    packages=find_packages(exclude=['contrib', 'doc', 'unit_tests']),
    install_requires=[],
    requires=info.REQUIRES,
    package_data={},
    data_files=[],
    entry_points={},
    scripts=glob.glob("scripts_python/*.py") + glob.glob("tests/*.py")
)
