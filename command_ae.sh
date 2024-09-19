experiments=experiments
experiment_name=fibercup_september24

rm -rf $experiments/$experiment_name

ae_train_model.py $experiments \
		   $experiment_name \
		   fibercup.hdf5 \
		   target \
		   -v INFO \
		   --batch_size_training 80 \
		   --batch_size_units nb_streamlines \
 	           --nb_subjects_per_batch 1 \
 	           --learning_rate 0.001 \
 	           --weight_decay 0.05 \
 	           --optimizer Adam \
 	           --max_epochs 1000 \
 	           --max_batches_per_epoch_training 20 \
	           --comet_workspace dwi_ml \
	           --comet_project ae-fibercup \
	           --patience 100 \
						 --use_gpu
