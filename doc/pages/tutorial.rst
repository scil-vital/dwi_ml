


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