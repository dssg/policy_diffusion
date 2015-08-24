cat /mnt/data/sunlight/dssg/model_legislation/extracted_model_legislation.json | parallel --pipe --delay 1.0 \
    --joblog /home/mburgess/model_legistlation_alignments.log \
    --tmpdir /mnt/data/sunlight/dssg/alignment_results/model_legislation_alignments \
    --files \
    /home/mburgess/policy_diffusion/scripts/generate_model_legislation_matches.py
