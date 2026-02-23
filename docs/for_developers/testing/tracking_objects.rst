.. _tracking:

Tracking with your model
========================

For tracking, you may observe how scripts `l2t_track_from_model` or `tt_track_from_model` work. They use two main objects, the Tracker and the Propagator, similarly as in scilpy.

Tracker
-------

toDO

Propagator
----------

toDo

Similarities with scilpy
------------------------

If you are familiar with scilpy, here is a comparison of our Tracker and Propagator to theirs:

**Similarities:**

- ToDo

**Differences:**

- In scilpy, the *theta* parameter defines an aperture cone inside which the next direction can be sampled. Here, sampling is not as straightforward. Ex, in the case of regression, the next direction is directly obtained from the model. Instead, theta is used as a stopping criterion.

- In scilpy, at each propagation step, the propagator uses the local model (ex, DTI, fODF) to decide the next direction. Here, the propagator sends data as input to the machine learning model. The model may receive additional inputs as compared to classical tractography (ex, the hidden states in RNNs, or the full beginning of the streamline in Transformers).

- GPU processing: As dwi_ml users tend to use GPU/CPU more than scilpy users, we offer a GPU options, where many streamlines are created simultaneously, to take advantage of the GPU capacities. In scilpy, CPU is always used, although possibly with parallel processes (not fully implemented yet).
