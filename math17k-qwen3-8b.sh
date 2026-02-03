project_name='Math-Qwen3-8B'
exp_name='RIPO'


model="/GenSIvePFS/users/models/Qwen3-8B-Base"
CKPTS_DIR=/GenSIvePFS/users/checkpoints/${project_name}/${exp_name}

train_bs=128
mini_bs=16
group_size=8
tensor_model_parallel_size=1
epochs=5

val_n=8
max_prompt_length=1024
max_response_length=15360

clip_ratio_low=0.05
clip_ratio_high=0.05
loss_agg_mode="token-mean"

python3 -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo \
 data.train_files=/GenSIvePFS/users/data/new/math-17k.parquet \
 data.val_files=/GenSIvePFS/users/data/new/aime24.parquet \
 data.train_batch_size=${train_bs} \
 data.max_prompt_length=${max_prompt_length} \
 data.max_response_length=${max_response_length} \
 data.filter_overlong_prompts=True \
 data.filter_overlong_prompts_workers=32 \
 data.truncation='error' \
 data.shuffle=True\
 actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
 actor_rollout_ref.model.path=${model} \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
 actor_rollout_ref.actor.ppo_mini_batch_size=${mini_bs} \
 actor_rollout_ref.actor.use_kl_loss=False \
 actor_rollout_ref.actor.kl_loss_coef=0 \
 actor_rollout_ref.actor.entropy_coeff=0 \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
 actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
 actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
 actor_rollout_ref.rollout.n=${group_size} \
 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
 actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=65536 \
 actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=65536 \
 actor_rollout_ref.actor.fsdp_config.param_offload=True \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
 actor_rollout_ref.ref.fsdp_config.param_offload=True \
 actor_rollout_ref.rollout.max_num_batched_tokens=96000 \
 actor_rollout_ref.actor.use_dynamic_bsz=True \
 actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
 actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
 actor_rollout_ref.rollout.enable_chunked_prefill=True \
 actor_rollout_ref.rollout.val_kwargs.n=${val_n} \
 algorithm.use_kl_in_reward=False \
 trainer.critic_warmup=0 \
 trainer.logger='["console","wandb"]' \
 trainer.project_name=${project_name} \
 trainer.experiment_name=${exp_name} \
 trainer.n_gpus_per_node=8 \
 trainer.nnodes=1 \
 trainer.val_before_train=True \
 trainer.save_freq=100 \
 trainer.test_freq=5 \
 trainer.default_local_dir="${CKPTS_DIR}" \
 trainer.total_epochs=${epochs}




