======
DWI_ML
======

.. image:: https://github.com/scil-vital/dwi_ml/workflows/test/badge.svg
  :alt:    Build Status

.. image:: https://readthedocs.org/projects/dwi-ml/badge/?version=latest
  :target: https://dwi-ml.readthedocs.io/en/latest/

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
  :target: https://github.com/scil-vital/dwi_ml/blob/master/LICENSE

Welcome to the `Sherbrooke Connectivity Imaging Lab (SCIL)`_ and
`Videos & Images Theory and Analytics Laboratory (VITAL)`_ joint DWI_ML
toolkit !

Links
=====

* `Getting started: installation and download <https://dwi-ml.readthedocs.io/en/latest/getting_started.html>`_
* `Contribute/Submit a patch <https://github.com/scil-vital/dwi_ml/blob/master/CONTRIBUTING.rst>`_
* `Issue tracking <https://github.com/scil-vital/dwi_ml/issues>`_

About
=====

DWI_ML is a toolkit for Diffusion Magnetic Resonance Imaging (dMRI) analysis
using machine learning and deep learning methods. It is mostly focused on the
tractography derivatives of dMRI.

**Here is the usual workflow for people using dwi_ml**:

1. `Organize your data <https://dwi-ml.readthedocs.io/en/latest/data_organization.html>`_. If your data is organized as expected, the following steps will be much easier.

2. `Preprocess your data <https://dwi-ml.readthedocs.io/en/latest/preprocessing.html>`_.
   - Preprocess your diffusion data using Tractoflow
   - Preprocess your tractogram using RecoBundlesX

3. `Create your own project with DWI_ML <https://dwi-ml.readthedocs.io/en/latest/processing.html>`_.
   - Create your own repository for your project.
   - Copy our scripts from ``please_copy_and_adapt``. Adapt based on your needs.
   - Train your ML algorithm
   - If needed, create your tracking algorithm based on your results.

License
=======

DWI_ML is licensed under the terms of the MIT license. Please see `LICENSE <./LICENSE>`_
file.

Citation
========

If you use DWI_ML in your dMRI data analysis, please cite the toolkit and
provide a link to it.


.. Links
.. Involved labs
.. _`Sherbrooke Connectivity Imaging Lab (SCIL)`: http://scil.dinf.usherbrooke.ca
.. _`Videos & Images Theory and Analytics Laboratory (VITAL)`: http://vital.dinf.usherbrooke.ca
