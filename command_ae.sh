experiments=experiments
experiment_name=mouse_september24

rm -rf $experiments/$experiment_name

ae_train_model.py $experiments \
		   $experiment_name \
		   mouse.hdf5 \
       target \
       -v INFO \
       --batch_size_training 2500 \
       --batch_size_units nb_streamlines \
       --nb_subjects_per_batch 5 \
       --learning_rate 0.0005*200 0.0003*200 0.0001*200 0.00007*200 0.00005 \
       --weight_decay 0.2 \
       --optimizer Adam \
       --max_epochs 2000 \
       --max_batches_per_epoch_training 9999 \
       --comet_workspace dwi-ml \
       --comet_project ae-fibercup \
       --patience 100 --use_gpu
