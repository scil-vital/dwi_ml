
Training and using: from start to finish
========================================

If you want, you can use our scripts to train our models with a new set of hyperparameters. No matter the model, the process will probably contain the following steps:

1. Creating a hdf5 file. Our library works with data in the hdf5 format. See :ref:`hdf5_usage` for more information.

2. Training the model. At each epoch, the script saves the model state if it is the best one so far, in a folder ``best_model``, but also always saves the model state and optimizer state in a checkpoint. This way, if anything happens and your training is stopped, you can continue training from the latest checkpoint.

3. Visualizing the logs to make sure you are satisfied with the results. For more information, see :ref:`visu_logs` for more information.

4. Using your newly trained model! For tractography models, this uses scripts such as ``**_track_from_model``. See :ref:`user_tracking` for more information.

Denoising models
----------------

Coming soon: Autoencoder (AE) model!

Tractography models
-------------------

Learn2track (l2t)
*****************

Full steps::

    # Create a hdf5 file
    dwiml_create_hdf5_dataset $input_folder $out_file $config_file \
        $training_subjs $validation_subjs $testing_subjs

    # Train a model. Play with options! Here are the mandatory inputs:
    l2t_train_model $saving_path $experiment_name $hdf5_file \
        $input_group_name $streamline_group_name

    # If you want to train your model a little more...
    l2t_resume_training_from_checkpoint $saving_path $experiment_name \
    --new_patience 10 --new_max_epochs 300

    # Visualize the logs
    dwiml_visualize_logs $saving_path

    # See which points of your validation streamlines have the worst loss
    l2t_visualise_loss $saving_path $hdf5_file $subj $input_group_name

    # Once happy, use your final model to track from it!
    l2t_track_from_model $saving_path $subj $input_group $out_tractgram $seeding_mask_group



TractographyTransformers (tt)
*****************************

Full steps::

    dwiml_create_hdf5_dataset $input_folder $out_file $config_file \
        $training_subjs $validation_subjs $testing_subjs

    tt_train_model ...

    tt_resume_training_from_checkpoint ...

    tt_track_from_model ...

    dwiml_visualize_logs ...
    tt_visualize_loss ...
    tt_visualize_weights ...

