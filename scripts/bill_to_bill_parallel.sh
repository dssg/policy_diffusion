cat /home/mburgess/policy_diffusion/data/bill_ids_random.txt | parallel --delay 0.1 \
    --joblog /home/mburgess/bill_to_bill_alignments.log \
    --tmpdir /mnt/data/sunlight/dssg/alignment_results/bill_to_bill_alignments \
    --files \
    /home/mburgess/policy_diffusion/scripts/generate_bill_to_bill_matches.py
