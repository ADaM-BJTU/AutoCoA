set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
# grpo
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/processed/hotpot_qa/train.parquet \
    data.val_files=data/processed/hotpot_qa/dev.parquet \
    data.train_batch_size=48 \
    data.val_batch_size=16 \
    data.shuffle=false \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    actor_rollout_ref.model.path=hf_checkpoint \
    actor_rollout_ref.actor.optim.lr=7e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    +actor_rollout_ref.rollout.max_search_nums=10 \
    actor_rollout_ref.rollout.name=vllm \
    +actor_rollout_ref.rollout.mode=search \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.disable_log_stats=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_example_hotpotqa' \
    trainer.experiment_name='stage2' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    +trainer.use_observation_mask=True \
    +trainer.val_before_train=False \
    trainer.save_freq=16 \
    trainer.test_freq=-1\
    trainer.val_generations_to_log_to_wandb=200\
    trainer.total_training_steps=96 \
    trainer.total_epochs=1 $@
    