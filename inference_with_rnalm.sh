# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export LAYERNORM_TYPE=fast_layernorm
export USE_DEEPSPEED_EVO_ATTTENTION=true
export CUTLASS_PATH=/lustre/scratch/shared-folders/bio_project/shuxian/rna_folding/cutlass   # for deepspeed
export HF_HOME=/lustre/scratch/shared-folders/bio_project/shuxian/gbft/hf_home

N_sample=5
N_step=200
N_cycle=10
seed=101
# use_deepspeed_evo_attention=true
input_json_path="./examples/test.json"
dump_dir="/lustre/scratch/shared-folders/bio_project/shuxian/rna_folding/protenix_output/output_msa_aido"
checkpoint_path="./output/protenix_finetune_msa_aidorna650m_bs16_len384_maxsteps10k_20250409_172311/checkpoints/3999_ema_0.999.pt"

python3 runner/inference.py \
--seeds ${seed} \
--dump_dir ${dump_dir} \
--input_json_path ${input_json_path} \
--model.N_cycle ${N_cycle} \
--sample_diffusion.N_sample ${N_sample} \
--sample_diffusion.N_step ${N_step} \
--augment.use_rnalm True \
--load_checkpoint_path ${checkpoint_path} \
--use_msa True