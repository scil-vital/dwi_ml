.. _visu_logs:

Visualizing logs
================

The trainer save a lot of information at each epoch: the training or validation loss in particular. It can send the information to Comet.ml on the fly; go discover their incredible website. Alternatively, you can run ``visualize_logs your_experiment`` to see the evolution of the losses and gradient norm.

Example of Comet.ml view:

        .. image:: /_static/images/example_comet.png
            :width: 1500