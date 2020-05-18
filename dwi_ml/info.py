"""This file contains defines parameters for DWI_ML that we use to fill
settings in `setup.py`, the DWI_ML top-level docstring, and for building the
docs.
"""

# DWI_ML version information. An empty _version_extra corresponds to a
# full release. '.dev' as a _version_extra string means this is a development
# version
_version_major = 0
_version_minor = 1
_version_micro = 0
_version_extra = 'dev'
# _version_extra = ''

# Format expected by setup.py and doc/conf.py: string of form "X.Y.Z"
__version__ = '%s.%s.%s%s' % (_version_major,
                              _version_minor,
                              _version_micro,
                              _version_extra)

classifiers = ['Development Status :: 3 - Alpha',
               'Intended Audience :: Science/Research',
               'License :: OSI Approved :: MIT License',
               'Programming Language :: Python :: 3',
               'Topic :: Scientific/Engineering',
               'Topic :: Scientific/Engineering ::Neuroimaging']

description = 'Diffusion Magnetic Resonance Imaging analysis toolkit ' + \
              'using machine learning and deep learning methods.'

keywords = 'dwi-ml DWI DL ML neuroimaging tractography'

# Main setup parameters
NAME = 'dwi_ml'
MAINTAINER = 'scil-vital'
MAINTAINER_EMAIL = ''
DESCRIPTION = description
URL = 'https://github.com/scil-vital/dwi_ml'
DOWNLOAD_URL = ''
BUG_TRACKER = 'https://github.com/scil-vital/dwi_ml/issues',
DOCUMENTATION = 'https://dwi-ml.readthedocs.io/en/latest/',
SOURCE_CODE = 'https://github.com/scil-vital/dwi_ml',
LICENSE = 'MIT license'
CLASSIFIERS = classifiers
KEYWORDS = keywords
AUTHOR = 'scil-vital'
AUTHOR_EMAIL = ''
PLATFORMS = ''
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
ISRELEASE = _version_extra == ''
VERSION = __version__
PROVIDES = ['dwi_ml']
REQUIRES = ['nibabel',
            'h5py']
