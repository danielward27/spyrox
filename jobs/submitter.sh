#!/bin/bash

# Example submission from spyrox root
# chmod +x jobs/submitter.sh && ./jobs/submitter.sh

# Define number of rounds and loss names
num_rounds=(1 2 4)
loss_names=("ELBO" "SNIS-fKL" "SoftCVI(a=0.75)" "SoftCVI(a=1)")

for loss in "${loss_names[@]}"; do
    for rounds in "${num_rounds[@]}"; do
        sbatch --job-name="${loss}_${rounds}" jobs/job.sh "$loss" "$rounds"
    done
done
