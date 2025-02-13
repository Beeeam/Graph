#!/bin/bash

#SBATCH -o ./slurm_result/slurm_test_%j.txt
#SBATCH -e ./slurm_result/slurm_error_%j.txt
#SBATCH --job-name=gp
#SBATCH --cpus-per-task=16
#SBATCH -p preempt --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --nodelist=bhg0059



source ~/.bashrc
source /software/miniconda3/4.12.0/bin/activate torch


python vaetrainer.py
# python vae_generate.py