#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=bio
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH -p gpumid
#SBATCH --exclusive
#SBATCH --exclude=gpumid-39,gpumid-53
#SBATCH --reservation=bio
#SBATCH --job-name rna_folding
#SBATCH --output=logs/_%j.out
#SBATCH --error=logs/_%j.err

export LAYERNORM_TYPE=fast_layernorm
export USE_DEEPSPEED_EVO_ATTTENTION=true
export CUTLASS_PATH=/lustre/scratch/users/Jiayou.Zhang/genbio/bytedance-protenix/cutlass
export HF_HOME=/lustre/scratch/shared-folders/bio_project/shuxian/gbft/hf_home
# wget -P /af3-dev/release_model/ https://af3-dev.tos-cn-beijing.volces.com/release_model/model_v0.2.0.pt
checkpoint_path="./release_data/checkpoint/model_v0.2.0.pt"


MODE="multi_gpu"  # single_gpu, multi_gpu, or multi_node
echo "Running in $MODE mode"


PROGRAM="./runner/train.py \
    --run_name protenix_finetune_msa_aidorna650m \
    --seed 42 \
    --base_dir ./output \
    --dtype bf16 \
    --project protenix \
    --use_wandb true \
    --wandb_entity shuxian-zou \
    --diffusion_batch_size 48 \
    --eval_interval 500 \
    --log_interval 50 \
    --checkpoint_interval 500 \
    --ema_decay 0.999 \
    --train_crop_size 384 \
    --max_steps 10000 \
    --warmup_steps 500 \
    --lr 0.001 \
    --sample_diffusion.N_step 20 \
    --load_checkpoint_path ${checkpoint_path} \
    --load_ema_checkpoint_path ${checkpoint_path} \
    --data.train_sets kaggle_train \
    --data.test_sets kaggle_test"


if [ $MODE == "single_gpu" ]; then
    python $PROGRAM
elif [ $MODE == "multi_gpu" ]; then
    torchrun --nproc-per-node 4 $PROGRAM
else
    export MASTER_PORT=5800
    nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
    nodes_array=($nodes)
    head_node=${nodes_array[0]}
    export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
    echo $MASTER_ADDR

    srun torchrun --nproc-per-node 4 \
    --nnodes $SLURM_NNODES \
    --node_rank $SLURM_PROCID \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    $PROGRAM
fi
