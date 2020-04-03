======
DWI_ML
======

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
  :target: https://github.com/scil-vital/dwi_ml/blob/master/LICENSE

Welcome to the `Sherbrooke Connectivity Imaging Lab (SCIL)`_ and
`Videos & Images Theory and Analytics Laboratory (VITAL)`_ joint DWI_ML
toolkit !

About
=====

DWI_ML is a toolkit for Diffusion Magnetic Resonance Imaging (dMRI) analysis
using machine learning and deep learning methods. It is mostly focused on the
tractography derivatives of dMRI.

**Here is the usual workflow for people using dwi_ml**:

    #. Preprocess your diffusion data using Tractoflow (see :ref:`ref_preprocessing`).
    #. Preprocess your tractogram using RecobundlesX (see :ref:`ref_preprocessing`).
    #. Create a new repository for your project. Create a 'scripts_bash' folder and copy our scripts from please_copy_and_adapt. Adapt based on your needs and run:

        #. Run organize_from_tractoflow.sh
        #. Run organize_from_recobundles.sh
        #. Run create_dataset.sh
        #. Run training.sh
        #. Run tracking.sh

.. toctree::
    :maxdepth: 1
    :caption: Table of contents

    getting_started
    preprocessing
    data_organization
    creating_hdf5

License
=======

DWI_ML is licensed under the terms of the MIT license. Please see `LICENSE <https://github.com/scil-vital/dwi_ml/blob/master/LICENSE>`_ file.

Citation
========

If you use DWI_ML in your dMRI data analysis, please cite the toolkit and
provide a link to it.

.. Links
.. Involved labs
.. _`Sherbrooke Connectivity Imaging Lab (SCIL)`: http://scil.dinf.usherbrooke.ca
.. _`Videos & Images Theory and Analytics Laboratory (VITAL)`: http://vital.dinf.usherbrooke.ca
.. _`Université de Sherbrooke`: https://www.usherbrooke.ca