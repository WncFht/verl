#!/bin/bash
set -x

# GPU设置
export CUDA_VISIBLE_DEVICES=1,2,3,4
export HF_HOME=/mnt/data2/huggingface
export TORCH_CUDA_ARCH_LIST=9.0
export HF_DATASETS_CACHE=$HOME/data/.cache/huggingface/datasets
export WANDB_MODE=offline

# 模型路径
MODEL=/mnt/data2/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3
PYTHONUNBUFFERED=1
TARGET_MODULES='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]'
# 如果使用vllm<=0.6.3，可能需要设置以下环境变量：
# export VLLM_ATTENTION_BACKEND=XFORMERS

# 首先运行数据预处理（如果需要）
# python3 /home/haotian/workspace/verl/recipe/tts-edit/tts_rm.py \
#     --data_dir /mnt/data2/datasets/tts-edit/edited_images/ \
#     --local_dir ~/data/tts_edit \
#     --train_split 0.9

# 运行PPO训练
python3 -m verl.trainer.main_ppo \
    data.train_files=$HOME/data/tts_edit_balanced/train.parquet \
    data.val_files=$HOME/data/tts_edit_balanced/test.parquet \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.target_modules=$TARGET_MODULES \
    actor_rollout_ref.model.exclude_modules='.*visual.*' \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.actor.optim.lr=3e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy="fsdp" \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    custom_reward_function.path=./recipe/tts-edit/tts_edit_reward.py \
    custom_reward_function.name=compute_score \
    critic.enable=False \
    reward_model.enable=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_tts_edit' \
    trainer.experiment_name='qwen2_5_vl_3b_tts_edit_scoring_balanced' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    trainer.rollout_data_dir=./record/rollout_generations \
    trainer.validation_data_dir=./record/validation_generations $@