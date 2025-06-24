#!/bin/bash

# ==============================================================================
# Final Hyperparameter Tuning for CLIP-MoRE
#
# This script focuses on optimizing around our new best result.
# Best Baseline so far: lr=8e-5, position=top1, params=qkv, rank=8
# ==============================================================================

# --- Helper function for colorful output ---
C_BLUE="\033[0;34m"
C_GREEN="\033[0;32m"
C_RESET="\033[0m"

# Function to announce the start of a new experiment
run_experiment() {
    echo -e "${C_BLUE}======================================================================${C_RESET}"
    echo -e "${C_GREEN}RUNNING EXPERIMENT:${C_RESET} $@"
    echo -e "${C_BLUE}======================================================================${C_RESET}"
    # NOTE: I've added a default for n_iters and alpha based on your last runs
    python3 main.py "$@"
    echo -e "${C_GREEN}Finished experiment.${C_RESET}\n"
}

# --- START OF FINAL EXPERIMENTS ---

# --- Series 1: Combine the Best LR with Other Promising Parameters ---
echo -e "\n\n${C_BLUE}### SERIES 1: Combining Best LR with Other Promising Modules ###${C_RESET}\n"

# Test 1.1: Your current best run used params=q,k,v. Another strong run used params=v,o.
# Let's see if the 'v,o' combo works even better with the new optimal learning rate.

run_experiment --lr 8e-5 --position top1 --params q k v --num_blocks 4 --block_rank 32 --dropout_rate 0.25 --alpha 1
run_experiment --lr 8e-5 --position top1 --params q k v --num_blocks 4 --block_rank 64 --dropout_rate 0.25 --alpha 1
run_experiment --lr 8e-5 --position top1 --params q k v --num_blocks 4 --block_rank 128 --dropout_rate 0.25 --alpha 1
run_experiment --lr 8e-5 --position top3 --params q k v --num_blocks 4 --block_rank 16 --dropout_rate 0.25 --alpha 1
run_experiment --lr 8e-5 --position top3 --params q k v --num_blocks 4 --block_rank 32 --dropout_rate 0.25 --alpha 1
run_experiment --lr 8e-5 --position top3 --params q k v --num_blocks 4 --block_rank 64 --dropout_rate 0.25 --alpha 1
run_experiment --lr 8e-5 --position top3 --params q k v --num_blocks 4 --block_rank 128 --dropout_rate 0.25 --alpha 1

run_experiment --lr 8e-5 --position up --params q k v --num_blocks 4 --block_rank 32 --dropout_rate 0.25 --alpha 1
run_experiment --lr 2e-4 --position up --params q k v --num_blocks 4 --block_rank 32 --dropout_rate 0.25 --alpha 1
run_experiment --lr 8e-5 --position up --params q k v --num_blocks 4 --block_rank 128 --dropout_rate 0.25 --alpha 1
run_experiment --lr 2e-4 --position up --params q k v --num_blocks 4 --block_rank 128 --dropout_rate 0.25 --alpha 1

run_experiment --lr 2e-4 --position all --params q k v --num_blocks 4 --block_rank 128 --dropout_rate 0.25 --alpha 1


echo -e "\n\n${C_GREEN}### ALL EXPERIMENTS COMPLETE ###${C_RESET}"
echo "Check testing_hyperparameters.txt for the final results."