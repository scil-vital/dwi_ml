#!/bin/bash

###############################################################################
# Your tree should look like:                                                 #
# derivatives                                                                 #
#    ├── original (ex, tractoflow input)                                      #
#    └── tractoflow_output                                                    #
#    └── dwi_ml_ready: will be created now.                                   #
#                                                                             #
# This script will create symlinks in dwi_ml_ready, pointing to data from     #
# tractoflow for each subject:                                                #
#       └── dwi:                                                              #
#           └── dwi: Resample/dwi_resample (last dwi output from tractoflow)  #
#           └── bval: Eddy/bval_eddy                                          #
#           └── bvec: Eddy/bvec_eddy                                          #
#           └── fa: DTI_metrics/fa                                            #
#       └── anat :                                                            #
#           └── t1_tractoflow: Register_T1/t1_warp                            #
#               (It is the last T1 output from tractoflow.)                   #
#       └── masks :                                                           #
#           └── wm: Segment_Tissues/mask_wm                                   #
#                                                                             #
# If you need something else for your model, you can modify this script.      #
#                                                                             #
# See our doc for more information                                            #
# (https://dwi-ml.readthedocs.io/en/latest/data_organization.html#ref-organization).
# We suppose that you have a "tractoflow_output" folder that contains RecobundlesX #
# results folder for each subject.                                            #
###############################################################################

# =====================================#
#  VARIABLES TO BE DEFINED BY THE USER #
# =====================================#
tractoflow_folder=$1   # Path to the working folder that contains the date
#                     (ex: database/derivatives/tractoflow).
dwi_ml_folder=$2     # Path where to save the output.
#                     (ex: database/derivatives/dwi_ml_ready)
subject_list=$3      # The list of subjects to arrange. A text file with one
#                      subject per line.

# =====================================#
#               CHECKS                 #
# =====================================#

####
# Cleaning paths
####
tractoflow_folder=$(realpath $tractoflow_folder)
dwi_ml_folder=$(realpath $dwi_ml_folder)

####
# Checking if paths exist
####
if [ ! -d $tractoflow_folder ]; then
    echo "Tractoflow output folder not found! ($tractoflow_folder)!"
    exit
fi
if [ ! -f $subject_list ]; then
    echo "Invalid subjects txt file! ($subject_list)"
    exit
fi

####
# Preparing dataset subfolders
####
if [ ! -d "$dwi_ml_folder" ]; then
    mkdir "$dwi_ml_folder"
else
    echo "The dwi_ml_ready folder already exists!!! Please delete it first."
    exit 1
fi


# =====================================#
#            MAIN SCRIPT               #
# =====================================#

echo "Checks passed. Now reorganizing subjects"
subjects=$(<$subject_list)
for subjid in $subjects
do
    echo "Reorganizing subject $subjid"
    mkdir $dwi_ml_folder/$subjid
    mkdir $dwi_ml_folder/$subjid/anat
    mkdir $dwi_ml_folder/$subjid/dwi
    mkdir $dwi_ml_folder/$subjid/masks
    mkdir $dwi_ml_folder/$subjid/wholebrain

    # dwi/dwi:
    if [ ! -f $tractoflow_folder/$subjid/Resample_DWI/${subjid}__dwi_resampled.nii.gz ]; then echo "Subject's DWI not found"; exit 1; fi
    ln -s $tractoflow_folder/$subjid/Resample_DWI/${subjid}__dwi_resampled.nii.gz $dwi_ml_folder/$subjid/dwi/dwi_tractoflow.nii.gz

    # dwi/bval
    if [ ! -f $tractoflow_folder/$subjid/Eddy/${subjid}__bval_eddy ]; then echo "Subject's bval not found"; exit 1; fi
    ln -s $tractoflow_folder/$subjid/Eddy/${subjid}__bval_eddy $dwi_ml_folder/$subjid/dwi/bval_tractoflow

    # dwi/bvec
    if [ ! -f $tractoflow_folder/$subjid/Eddy/${subjid}__dwi_eddy_corrected.bvec ]; then echo "Subject's bvec not found"; exit 1; fi
    ln -s $tractoflow_folder/$subjid/Eddy/${subjid}__dwi_eddy_corrected.bvec $dwi_ml_folder/$subjid/dwi/bvec_tractoflow

    # dwi/FA
    if [ ! -f $tractoflow_folder/$subjid/DTI_Metrics/${subjid}__fa.nii.gz ]; then echo "Subject's FA not found"; exit 1; fi
    ln -s $tractoflow_folder/$subjid/DTI_Metrics/${subjid}__fa.nii.gz $dwi_ml_folder/$subjid/dwi/fa.nii.gz

    # anat/T1:
    if [ ! -f $tractoflow_folder/$subjid/Register_T1/${subjid}__t1_warped.nii.gz ]; then echo "Subject's T1 not found"; exit 1; fi
    ln -s $tractoflow_folder/$subjid/Register_T1/${subjid}__t1_warped.nii.gz $dwi_ml_folder/$subjid/anat/t1_tractoflow.nii.gz

    # mask/wm
    if [ ! -f $tractoflow_folder/$subjid/Segment_Tissues/${subjid}__map_wm.nii.gz ]; then echo "Subject's WM map not found"; exit 1; fi
    ln -s $tractoflow_folder/$subjid/Segment_Tissues/${subjid}__map_wm.nii.gz $dwi_ml_folder/$subjid/masks/wm.nii.gz

    # tractograms:
    # Modify this based on your tractogram's name. Or skip and use organaize_from_recobundles.sh.
    #if [ ! -f $tractoflow_folder/$subjid/*Tracking/${subjid}__*tracking*.trk ]; then echo "Subject's tractogram not found!"; exit 1; fi
    #ln -s $tractoflow_folder/$subjid/*Tracking/${subjid}__*tracking*.trk $dwi_ml_folder/$subjid/bundles/tractoflow_wholebrain.trk

done

echo "We have organized tractoflow results from processed into dwi_ml_ready (dwi, anat, masks)"
