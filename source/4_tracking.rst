4. Using your model (ex: Tracking)
==================================

General testing of a model
--------------------------

This step depends on your model and your choice of metrics, but in generative models, you probably want to track on new data and verify the quality of your reconstruction. We have prepared a script that allows you to track from a model.

Tracking with a model
---------------------

The script track_from_model.py has been prepared in the please_copy_and_adapt folder. It uses classes Tracker and Propagator, similarly as in scilpy.

**Similarities:**

- The propagator uses Runge-Kutta integration of order 1, 2 or 4 to choose the next direction.

- If no acceptable/valid direction are found, the propagation advances straight ahead, for a maximum of continuous invalid directions given by the user.

**Differences:**

- In scilpy, the *theta* parameter defines an aperture cone inside which the next direction can be sampled. Here, sampling is not as straightforward. Ex, in the case of regression, the next direction is directly obtained from the model. Instead, theta is used as a stopping criterion.

- In scilpy, at each propagation step, the data is used directly to get a direction. Here, the data is used as input to the model. This means the model is ran at each step of the Runge-Kutta integration.

- (upcoming): As dwi_ml users tend to use GPU/CPU more than scilpy users, a different implementation should be coded soon, where many streamlines are created simultaneously, to take advantage of the GPU capacities. In scilpy, CPU is always used, although possibly with parallel processes.
