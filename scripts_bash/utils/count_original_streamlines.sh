###
# Script to count the number of original streamlines
# Args:
#   dataset_folder : Path to the folder containing the original/ dataset folder
#   all_subjects_txt : .txt file that contains the list of all subjects
###
dataset_folder=$1
all_subjects_txt=$2

total=0
for subject in $(<"$all_subjects_txt"); do
  for tract in "$dataset_folder"/original/bundles/"$subject"/tracts/*.trk; do
    bundlename=$(basename "$tract")
    bundlename=${bundlename%.trk}
    echo "Computing # of streamlines for subject $subject and bundle $bundlename..."
    json_result=$(scil_count_streamlines.py "$tract")
    echo "Result: $json_result"
    found=$(echo "$json_result" | python3 -c "import sys, json; print(json.load(sys.stdin)['$bundlename']['tract_count'])")
    total=$((total + found))
    echo "Found $found streamlines; total: $total"
  done
done
echo "Total number of streamlines : $total"
