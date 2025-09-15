#!/bin/bash
set -x

# GPU设置
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=/mnt/data2/huggingface
export TORCH_CUDA_ARCH_LIST=9.0
export WANDB_MODE=offline

# 模型路径
MODEL=/mnt/data2/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3
export PYTHONUNBUFFERED=1

# 如果使用vllm<=0.6.3，可能需要设置以下环境变量：
# export VLLM_ATTENTION_BACKEND=XFORMERS

# 首先运行数据预处理（如果需要）
# python3 /home/haotian/workspace/verl/recipe/tts-edit/tts_rm.py \
#     --data_dir /mnt/data2/datasets/tts-edit/edited_images/ \
#     --local_dir ~/data/tts_edit \
#     --train_split 0.9

# 运行PPO训练
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/tts_edit/train.parquet \
    data.val_files=$HOME/data/tts_edit/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.lora_rank=128 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.exclude_modules='.*visual.*' \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy="fsdp" \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.max_tokens=512 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    custom_reward_function.path=/home/haotian/workspace/verl/recipe/tts-edit/tts_edit_reward.py \
    custom_reward_function.name=compute_score \
    reward_model.enable=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_tts_edit' \
    trainer.experiment_name='qwen2_5_vl_3b_tts_edit_scoring' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    trainer.rollout_data_dir=./rollout_generations \
    trainer.validation_data_dir=./validation_generations $@