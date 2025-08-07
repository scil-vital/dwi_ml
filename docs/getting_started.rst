Getting started: download and installation
==========================================

Downloading dwi_ml
******************

To use the DWI_ML toolkit you will need to clone the repository and install the required dependencies::

   git clone https://github.com/scil-vital/dwi_ml.git

Installing dependencies
***********************

We support python 3.11.  (python3.11-distutils and python3.11-dev must also be installed).

The toolkit relies on `Scilpy`_ and on a number of other packages available through the `Python Package Index (PyPI)`_ (i.e. you can use pip).

We strongly recommend working in a virtual environment to install all dependencies related to DWI_ML.

- The toolkit heavily relies on deep learning methods. As such, a GPU device will be instantiated whenever one is available. DWI_ML uses PyTorch as its deep learning back end. Thus, in order to use DWI_ML deep learning methods you will need to take a few additional steps.

**Cuda**:

  - Verify that your computer has the required capabilities in the *Pre-installation Actions* section at `cuda/cuda-installation-guide <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`_ (sections 2.1 - 2.4). To find your OS version and the available GPU, check the *About* menu in your computer settings.

  - Follow the download instructions at `nvidia.com/cuda-downloads <https://developer.nvidia.com/cuda-downloads>`_. Choose the environment that fits your system in the selector. You can choose *deb(local)* for the installer type.

  - Follow the installation instructions.

**Creating a Comet account**:

- The toolkit uses `comet_ml <https://www.comet.ml/docs/python-sdk/advanced/>`_. It is a python library that creates an "Experiment" (ex, training a model with a given set of hyperparameters) which automatically creates many types of logs online. It requires user to set an API key in $HOME/.comet.config with contents:

        | [comet]
        | api_key=YOUR-API-KEY

Alternatively, you can add it as an environment variable. Add this to your $HOME/.bashrc file.

        | export COMET_API_KEY=YOUR-API-KEY

  An API (application programming interface) is a code that gets passed in by applications, containing information to identify its user, for instance. To get an API key, see `<https://https://www.comet.com/docs/v2/guides/getting-started/quickstart/#get-an-api-key>`_. Click on the key icon, copy value to the clipboard and save it in your file in $HOME.


Installing dwi_ml
*****************

If you want to install the toolkit on your machine or your virtual environment, as a user you should type::

   export SETUPTOOLS_USE_DISTUTILS=stdlib
   pip install uv
   uv pip install .

Or, if you want to develop, use::

   export SETUPTOOLS_USE_DISTUTILS=stdlib
   pip install uv
   uv pip install -e .

.. Links
.. Python-related tools
.. _`Python Package Index (PyPI)`: https://pypi.org

.. Toolkits/packages
.. _CUDA: https://developer.nvidia.com/cuda-zone
.. _VITALabAi: https://bitbucket.org/vitalab/vitalabai_public
.. _Scilpy: https://github.com/scilus/scilpy
