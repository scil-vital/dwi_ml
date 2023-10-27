
Batch sampler
=============

These classes defines how to sample the streamlines available in the
MultiSubjectData.

**AbstractBatchSampler:**

- Defines the __iter__ method:

    - Finds a list of streamlines ids and associated subj that you can later load in your favorite way.

- Define the load_batch method:

    - Loads the streamlines associated to sampled ids. Can resample them.

    - Performs data augmentation (on-the-fly to avoid having to multiply data on disk) (ex: splitting, reversing, adding noise).

Child class : **BatchStreamlinesSamplerOneInput:**

- Redefines the load_batch method:

    - Now also loads the input data under each point of the streamline (and possibly its neighborhood), for one input volume.

You are encouraged to contribute to dwi_ml by adding any child class here.
