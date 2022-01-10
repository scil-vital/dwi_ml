5. Preparing your model
=======================

You should make your model a child class of our **MainModelAbstract** to keep some important properties (ex, experiment name, neighborhood definition). Also, methods to save and load the model parameters on disk have been prepared.

The compute_loss method should be implemented to be used with our trainer.

For generative models, the get_tracking_direction_det and sample_tracking_direction_prob methods should be implemented to be used with our tracker.

We have also prepared child classes to help formatting previous directions, useful both for training and tracking.