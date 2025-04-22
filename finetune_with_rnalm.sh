export LAYERNORM_TYPE=fast_layernorm
export USE_DEEPSPEED_EVO_ATTTENTION=true
export CUTLASS_PATH=/lustre/scratch/shared-folders/bio_project/shuxian/rna_folding/cutlass   # for deepspeed
export HF_HOME=/lustre/scratch/shared-folders/bio_project/shuxian/gbft/hf_home

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL


MODE="multi_gpu"  # single_gpu, multi_gpu, or multi_node
echo "Running in $MODE mode"


# wget -P /af3-dev/release_model/ https://af3-dev.tos-cn-beijing.volces.com/release_model/model_v0.2.0.pt
checkpoint_path="./release_data/checkpoint/model_v0.2.0.pt"


run_name=protenix_finetune_aidorna1.6b_bs16_len640_lr5e-4_maxsteps10k

PROGRAM="./runner/train.py \
    --run_name ${run_name} \
    --seed 42 \
    --base_dir ./output \
    --dtype bf16 \
    --project protenix \
    --use_wandb True \
    --wandb_entity shuxian-zou \
    --diffusion_batch_size 32 \
    --eval_first True \
    --eval_ema_only True \
    --iters_to_accumulate 4 \
    --eval_interval 400 \
    --log_interval 10 \
    --checkpoint_interval 400 \
    --ema_decay 0.999 \
    --train_crop_size 640 \
    --test_max_n_token 1024 \
    --max_steps 10000 \
    --warmup_steps 200 \
    --lr 5e-4 \
    --augment.use_rnalm True \
    --augment.rnalm_name aido_rna_1b600m \
    --sample_diffusion.N_step 20 \
    --load_checkpoint_path ${checkpoint_path} \
    --load_ema_checkpoint_path ${checkpoint_path} \
    --data.train_sets kaggle_train \
    --data.test_sets kaggle_test \
    --data.msa.enable_prot_msa False \
    --data.msa.enable_rna_msa True
    "


if [ $MODE == "single_gpu" ]; then
    LOG="logs/${run_name}_$(date +%Y%m%d_%H%M%S).log"
    CUDA_VISIBLE_DEVICES=1 python $PROGRAM 2>&1 | tee $LOG

elif [ $MODE == "multi_gpu" ]; then
    LOG="logs/${run_name}_$(date +%Y%m%d_%H%M%S).log"
    torchrun --nproc-per-node 4 $PROGRAM 2>&1 | tee $LOG

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
