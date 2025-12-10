# HCC-ClinReasoner

A clinically aligned large language model specialized for hepatocellular carcinoma (HCC).  
The model takes real-world Chinese or English EMR narrative text as input and directly outputs:

- Continuous risk score and corresponding risk-based stage  
- Ranked Top-3 guideline-consistent treatment recommendations with traceable evidence and Chain-of-Thought reasoning  
- Precise month-level survival prediction (1/3/5-year survival probability and median OS)

## Repository Status
- Paper: under review  
- Model weights: will be released after paper acceptance and IRB approval  
- Training data: cannot be publicly released due to privacy regulations  


## Environment Setup

- **SFT stage**: Please follow the official installation instructions of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- **RL stage**: Please follow the official installation instructions of [VERL](https://github.com/volcengine/verl)

## 1. Data Generation


## 2. Model Training (Two-Stage)

### Stage 1: Supervised Fine-Tuning (SFT)  
Base model: Qwen3-32B  
Framework: LLaMA-Factory  

```bash
# scripts/train_sft.sh
#!/bin/bash
# 8× A100/H100 (80GB) recommended
FORCE_TORCHRUN=1 llamafactory-cli train configs/hcc-clinreasoner-sft.yaml
```
The YAML configuration files under `config` contain detailed comments and are ready to use.


### Stage 2: Experience-Accrual Reinforcement Learning (EARL)
Base model: hcc-clinreasoner-sft  
Framework: VERL 


Reward: clinician-designed composite reward (guideline consistency + evidence quality + safety + calibration)



```bash
# scripts/train_rl.sh
#!/bin/bash
# 8× B200 required

export PATH="xxx/miniconda3/envs/verl_v1/bin:$PATH"
source activate xxx/miniconda3/envs/verl_v1

set -x

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    custom_reward_function.path=verl/utils/reward_score/hcc_reasoner.py \
    data.train_files=data/rl.parque \
    data.val_files=data/val_internal.parque \
    data.train_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=saved_models/hcc-clinreasoner-sft \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True\
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24576 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=49152 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=20 \
    +actor_rollout_ref.rollout.repetition_penalty=1.05\
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=-0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_liver' \
    trainer.experiment_name='hcc_clinreasoner' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=5 \
    trainer.default_local_dir=saved_models/HCC_ClinReasoner \
    trainer.total_epochs=3 $@
```



## 3. Model evaluation
You first need to generate model outputs (json format) by the following step:

Model merge:
```
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir checkpoints/<project_name>/<experiment_name>/global_step_X/actor \
  --target_dir /path/to/merged_hf_model

```

Launch a vLLM session for response generation:
```
vllm serve /path/to/merged_hf_model  --tensor-parallel-size 8
```

Generate predictions of the internal SEER test set:
```
cd scripts
python test_seer_vllm_tts.py --model '/path/to/merged_hf_model' --output '/path/to/seer_outputs'
```

Generate predictions of the external test set:
```
python test_chunggeng_vllm_tts.py --model '/path/to/merged_hf_model' --output '/path/to/changguang_outputs'
```

```
python test_chunggeng_vllm_multi_center_tts.py --model '/path/to/merged_hf_model' --output '/path/to/other_center_outputs'
```

<!-- ## 4. Visualization -->


