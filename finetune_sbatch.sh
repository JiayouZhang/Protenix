#!/bin/bash -l
#SBATCH --nodes=2
#SBATCH --account=bio
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH -p gpumid
#SBATCH --exclusive
#SBATCH --exclude=gpumid-53,gpumid-41
#SBATCH --reservation=bio
#SBATCH --job-name rna_task
#SBATCH --output=logs/protenix_finetune_msa_aidorna650m_bs16_%j.out
#SBATCH --error=logs/protenix_finetune_msa_aidorna650m_bs16_%j.err

srun bash finetune_with_rnalm.sh