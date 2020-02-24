======
DWI_ML
======

.. image:: https://github.com/scil-vital/dwi_ml/workflows/CI%20testing/badge.svg?event=push
  :target: https://github.com/scil-vital/dwi_ml/actions?query=workflow%3ABuild+branch%3Amaster

.. image:: https://readthedocs.org/projects/dwi-ml/badge/?version=latest
  :target: https://dwi-ml.readthedocs.io/en/latest/

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
  :target: https://github.com/scil-vital/dwi_ml/blob/master/LICENSE

Welcome to the `Sherbrooke Connectivity Imaging Lab (SCIL)`_ and
`Videos & Images Theory and Analytics Laboratory (VITAL)`_ joint DWI_ML
toolkit !

Links
=====

* `Getting started <./doc/getting_started.rst>`_
* `Submit a patch <./CONTRIBUTING.rst>`_
* `Issue tracking <https://github.com/scil-vital/dwi_ml/issues>`_

About
=====

DWI_ML is a toolkit for Diffusion Magnetic Resonance Imaging (dMRI) analysis
using machine learning and deep learning methods. It is mostly focused on the
tractography derivatives of dMRI.

**Here is the usual workflow for people using dwi_ml**:

#. Preprocess your diffusion data using Tractoflow (see :ref:``ref_preprocessing``).
#. Preprocess your tractogram using RecoBundlesX (see :ref:``ref_preprocessing``).
#. Create a new repository for your project. Create a ``scripts_bash`` folder
   and copy our scripts from ``please_copy_and_adapt``. Adapt based on your
   needs and run:

 #. ``organize_from_tractoflow.sh``
 #. ``organize_from_recobundles.sh``
 #. ``create_dataset.sh``
 #. ``training.sh``
 #. ``tracking.sh``

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
