export LAYERNORM_TYPE=fast_layernorm
export USE_DEEPSPEED_EVO_ATTTENTION=true
export CUTLASS_PATH=/lustre/scratch/users/Jiayou.Zhang/genbio/bytedance-protenix/cutlass
export HF_HOME=/lustre/scratch/shared-folders/bio_project/shuxian/gbft/hf_home
# wget -P /af3-dev/release_model/ https://af3-dev.tos-cn-beijing.volces.com/release_model/model_v0.2.0.pt
checkpoint_path="./release_data/checkpoint/model_v0.2.0.pt"


MODE="multi_node"  # single_gpu, multi_gpu, or multi_node
echo "Running in $MODE mode"


PROGRAM="./runner/train.py \
    --run_name protenix_finetune_msa_aidorna650m_bs16 \
    --seed 42 \
    --base_dir ./output \
    --dtype bf16 \
    --project protenix \
    --use_wandb true \
    --wandb_entity shuxian-zou \
    --diffusion_batch_size 32 \
    --eval_interval 500 \
    --log_interval 50 \
    --checkpoint_interval 500 \
    --ema_decay 0.999 \
    --train_crop_size 640 \
    --max_steps 50000 \
    --warmup_steps 500 \
    --lr 0.001 \
    --sample_diffusion.N_step 20 \
    --load_checkpoint_path ${checkpoint_path} \
    --load_ema_checkpoint_path ${checkpoint_path} \
    --data.train_sets kaggle_train \
    --data.test_sets kaggle_recentPDB,kaggle_test
    "


if [ $MODE == "single_gpu" ]; then
    python $PROGRAM
elif [ $MODE == "multi_gpu" ]; then
    torchrun --nproc-per-node 4 $PROGRAM
else

    MASTER_PORT=55863
    MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    echo "MASTER_ADDR=$MASTER_ADDR  NODE_RANK=$SLURM_NODEID"

    torchrun --nproc-per-node 4 \
    --nnodes $SLURM_NNODES \
    --node_rank $SLURM_NODEID \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    $PROGRAM
fi
