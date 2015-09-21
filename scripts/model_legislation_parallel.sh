$model_legislation_path="/mnt/data/sunlight/dssg/model_legislation/extracted_model_legislation.json"
$job_log_path="/home/mburgess/model_legistlation_alignments.log"
$temp_dir_path="/mnt/data/sunlight/dssg/alignment_results/model_legislation_alignments"
$script_path="/home/mburgess/policy_diffusion/scripts/generate_model_legislation_matches.py"

cat $model_legislation_path | parallel --pipe --delay 1.0 --joblog $job_log_path \
    --tmpdir $temp_dir_path --files  $script_path
