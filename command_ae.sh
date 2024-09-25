experiments=experiments
experiment_name=fibercup_september24

rm -rf $experiments/$experiment_name

ae_train_model.py $experiments \
		   $experiment_name \
       fibercup_tracking.hdf5 \
       target \
       -v INFO \
       --batch_size_training 1800 \
       --batch_size_units nb_streamlines \
       --nb_subjects_per_batch 5 \
       --learning_rate 0.00001*300 0.000005 \
       --weight_decay 0.2 \
       --optimizer Adam \
       --max_epochs 5000 \
       --max_batches_per_epoch_training 9999 \
       --comet_workspace dwi-ml \
       --comet_project ae-fibercup \
       --patience 100 --use_gpu
