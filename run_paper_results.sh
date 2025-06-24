#!/bin/bash

# ==============================================================================
# Final Hyperparameter Tuning for CLIP-MoRE
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
    python3 main.py "$@"
    echo -e "${C_GREEN}Finished experiment.${C_RESET}\n"
}

# --- START OF FINAL EXPERIMENTS ---
run_experiment --lr 8e-5 --position top3 --params q k v --num_blocks 4 --block_rank 128 --dropout_rate 0.25 --alpha 1 --shots 16
run_experiment --lr 8e-5 --position top3 --params q k v --num_blocks 4 --block_rank 128 --dropout_rate 0.25 --alpha 1 --shots 16
run_experiment --lr 8e-5 --position top3 --params q k v --num_blocks 4 --block_rank 128 --dropout_rate 0.25 --alpha 1 --shots 16

run_experiment --lr 8e-5 --position top3 --params q k v --num_blocks 4 --block_rank 128 --dropout_rate 0.25 --alpha 1 --shots 4
run_experiment --lr 8e-5 --position top3 --params q k v --num_blocks 4 --block_rank 128 --dropout_rate 0.25 --alpha 1 --shots 4
run_experiment --lr 8e-5 --position top3 --params q k v --num_blocks 4 --block_rank 128 --dropout_rate 0.25 --alpha 1 --shots 4

run_experiment --lr 8e-5 --position top3 --params q k v --num_blocks 4 --block_rank 128 --dropout_rate 0.25 --alpha 1 --shots 1
run_experiment --lr 8e-5 --position top3 --params q k v --num_blocks 4 --block_rank 128 --dropout_rate 0.25 --alpha 1 --shots 1
run_experiment --lr 8e-5 --position top3 --params q k v --num_blocks 4 --block_rank 128 --dropout_rate 0.25 --alpha 1 --shots 1

echo -e "\n\n${C_GREEN}### ALL EXPERIMENTS COMPLETE ###${C_RESET}"