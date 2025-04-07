
export LAYERNORM_TYPE=fast_layernorm
export USE_DEEPSPEED_EVO_ATTTENTION=true
export CUTLASS_PATH=/lustre/scratch/users/Jiayou.Zhang/genbio/bytedance-protenix/cutlass
# wget -P /af3-dev/release_model/ https://af3-dev.tos-cn-beijing.volces.com/release_model/model_v0.2.0.pt
checkpoint_path="/home/jiayou.zhang/hom/personal/rna-stanford/kaggle/Protenix/release_data/checkpoint/model_v0.2.0.pt"

python ../Protenix/runner/train.py \
--run_name protenix_finetune \
--seed 42 \
--base_dir ./output_demo \
--dtype bf16 \
--project protenix \
--use_wandb false \
--diffusion_batch_size 48 \
--eval_interval 400 \
--log_interval 50 \
--checkpoint_interval 400 \
--ema_decay 0.999 \
--train_crop_size 384 \
--max_steps 100000 \
--warmup_steps 2000 \
--lr 0.001 \
--sample_diffusion.N_step 20 \
--load_checkpoint_path ${checkpoint_path} \
--load_ema_checkpoint_path ${checkpoint_path} \
--data.train_sets kaggle_train \
--data.test_sets kaggle_test