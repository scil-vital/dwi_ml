import os

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

with open('requirements.txt') as f:
    required_dependencies = f.read().splitlines()
    external_dependencies = []
    for dependency in required_dependencies:
        if dependency[0:2] == '-e':
            repo_name = dependency.split('=')[-1]
            repo_url = dependency[3:]
            external_dependencies.append('{} @ {}'.format(repo_name, repo_url))
        else:
            external_dependencies.append(dependency)


# Get version and release info, which is all stored in version.py
ver_file = os.path.join('dwi_ml', 'version.py')
with open(ver_file) as f:
    exec(f.read())
opts = dict(name=NAME,
            maintainer=MAINTAINER,
            maintainer_email=MAINTAINER_EMAIL,
            description=DESCRIPTION,
            long_description=LONG_DESCRIPTION,
            url=URL,
            download_url=DOWNLOAD_URL,
            license=LICENSE,
            classifiers=CLASSIFIERS,
            author=AUTHOR,
            author_email=AUTHOR_EMAIL,
            platforms=PLATFORMS,
            version=VERSION,
            packages=find_packages(),
            python_requires=PYTHON_VERSION,
            setup_requires=['numpy'],
            install_requires=external_dependencies,
            entry_points={
                'console_scripts': ["{}=scripts_python.{}:main".format(
                    os.path.basename(s),
                    os.path.basename(s).split(".")[0]) for s in PYTHON_SCRIPTS]
            },
            scripts=[s for s in BASH_SCRIPTS],
            data_files=[],
            include_package_data=True)

setup(**opts)
