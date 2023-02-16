#!/bin/bash

###############################################################################
# Your tree should look like:                                                 #
# derivatives                                                                 #
#    └── tractoflow_output                                                    #
#    └── ...
#    └── dwi_ml_ready: will be created now.                                   #
#                                                                             #
# This script will create symlinks in dwi_ml_ready, pointing to data from     #
# tractoflow for each subject:                                                #
#       └── dwi:                                                              #
#           └── dwi: Resample/dwi_resample (last dwi output from tractoflow)  #
#           └── fodf: FODF_Metrics/fodf                                       #
#           └── bval: Eddy/bval_eddy                                          #
#           └── bvec: Eddy/bvec_eddy                                          #
#           └── fa: DTI_metrics/fa                                            #
#       └── anat :                                                            #
#           └── t1_tractoflow: Register_T1/t1_warped                          #
#               (It is the last T1 output from tractoflow.)                   #
#       └── masks :                                                           #
#           └── wm: Segment_Tissues/mask_wm                                   #
#       └── bundles  :                                                        #
#           └── *: Tracking*/*.trk                                            #
#               or Tracking*/*/*.trk                                          #
#               (names become Tracking*_*_*.trk)                              #
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
tractoflow_folder=$1   # Path to the working folder that contains the data
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
    mkdir $dwi_ml_folder/$subjid/tractograms

    # dwi/dwi:
    echo "  File: dwi_resampled"
    dwi=$tractoflow_folder/$subjid/Resample_DWI/${subjid}__dwi_resampled.nii.gz
    if [ ! -f $dwi ]; then echo "Subject's DWI not found: $dwi"; exit 1; fi
    ln -s $dwi $dwi_ml_folder/$subjid/dwi/dwi.nii.gz

    # dwi/fodf:
    echo "  File: fodf"
    fODF=$tractoflow_folder/$subjid/FODF_Metrics/${subjid}__fodf.nii.gz
    if [ ! -f $fODF ]; then echo "Subject's fODF not found: $fODF"; exit 1; fi
    ln -s $fODF $dwi_ml_folder/$subjid/dwi/fodf.nii.gz

    # dwi/bval
    echo "  File: bval"
    bval=$tractoflow_folder/$subjid/Eddy/${subjid}__bval_eddy
    if [ ! -f $bval ]; then
        bval=$tractoflow_folder/$subjid/Eddy_Topup/${subjid}__bval_eddy
        if [ ! -f $bval ]; then echo "Subject's bval not found: $bval"; exit 1; fi
    fi
    ln -s $bval $dwi_ml_folder/$subjid/dwi/bval

    # dwi/bvec
    echo "  File: bvec"
    bvec=$tractoflow_folder/$subjid/Eddy/${subjid}__dwi_eddy_corrected.bvec
    if [ ! -f $bvec ]; then
        bvec=$tractoflow_folder/$subjid/Eddy_Topup/${subjid}__dwi_eddy_corrected.bvec
        if [ ! -f $bvec ]; then echo "Subject's bvec not found: $bvec"; exit 1; fi
    fi
    ln -s $bvec $dwi_ml_folder/$subjid/dwi/bvec.bvec

    # dwi/FA
    echo "  File: fa"
    fa=$tractoflow_folder/$subjid/DTI_Metrics/${subjid}__fa.nii.gz
    if [ ! -f $fa ]; then echo "Subject's FA not found: $fa"; exit 1; fi
    ln -s $fa $dwi_ml_folder/$subjid/dwi/fa.nii.gz

    # anat/T1:
    echo "  File: t1"
    t1=$tractoflow_folder/$subjid/Register_T1/${subjid}__t1_warped.nii.gz
    if [ ! -f $t1 ]; then echo "Subject's T1 not found: $t1"; exit 1; fi
    ln -s $t1 $dwi_ml_folder/$subjid/anat/t1.nii.gz

    # mask/wm
    echo "  File: wm"
    wm=$tractoflow_folder/$subjid/Segment_Tissues/${subjid}__mask_wm.nii.gz
    if [ ! -f $wm ]; then echo "Subject's WM map not found: $wm"; exit 1; fi
    ln -s $wm $dwi_ml_folder/$subjid/masks/wm.nii.gz

    # tractograms:
    echo "  Tractograms:"
    tracking_files=$(ls $tractoflow_folder/$subjid/Tracking*/*.trk 2>/dev/null)  # 2>... : to suppress error "No such file or directory"
    other_files=$(ls $tractoflow_folder/$subjid/Tracking*/*/*.trk 2>/dev/null)
    for tracking_file in $tracking_files $other_files
    do
        filename=${tracking_file#$tractoflow_folder/$subjid/}
        echo "    Found: $filename"  # File contains the whole path.
        linkname=$(echo $filename | tr / _)
        echo "       Link name: $linkname"
        ln -s $tracking_file $dwi_ml_folder/$subjid/tractograms/$linkname
    done

done

echo "We have organized tractoflow results from processed into dwi_ml_ready (dwi, anat, masks)"
