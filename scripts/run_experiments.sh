#!/bin/bash
# ============================================================
# run_experiments.sh — Full NSM Experiment Pipeline
# ============================================================
# Run all experiments for the paper:
#   1. Baseline (original binary reward)
#   2. NSM reward (our method)
#   3. NSM + Warm-start
#   4. NSM + Annealing
#   5. Ablations
#
# Prerequisites:
#   - TinyZero repo cloned and installed (pip install -e .)
#   - Models downloaded (Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-3B)
#   - This project's files copied into TinyZero
#
# Usage:
#   bash run_experiments.sh [experiment_name]
#   bash run_experiments.sh all
# ============================================================

set -e

# ======================== CONFIG ========================
export PROJ_DIR=${PROJ_DIR:-$(pwd)}
export DATA_DIR=${DATA_DIR:-$PROJ_DIR/data}
export CKPT_DIR=${CKPT_DIR:-$PROJ_DIR/checkpoints}
export VLLM_ATTENTION_BACKEND=XFORMERS

# Models — set these to your local paths
export MODEL_05B=${MODEL_05B:-"Qwen/Qwen2.5-0.5B"}
export MODEL_15B=${MODEL_15B:-"Qwen/Qwen2.5-1.5B"}
export MODEL_3B=${MODEL_3B:-"Qwen/Qwen2.5-3B"}

# Reward function path
export NSM_REWARD_PATH=${NSM_REWARD_PATH:-"$PROJ_DIR/reward_functions/countdown_nsm.py"}

# ======================== HELPERS ========================

prepare_data() {
    echo "========================================"
    echo "Step 1: Preparing datasets"
    echo "========================================"
    
    # Original TinyZero dataset (baseline)
    python $PROJ_DIR/data_preprocess/countdown_multidiff.py \
        --local_dir $DATA_DIR/original \
        --template_type base \
        --original_mode \
        --train_size 327680 \
        --test_size 1024
    
    # Multi-difficulty dataset
    python $PROJ_DIR/data_preprocess/countdown_multidiff.py \
        --local_dir $DATA_DIR/multidiff \
        --template_type base \
        --difficulties easy medium hard \
        --train_size 300000 \
        --test_size 1024 \
        --warmstart_size 100
    
    echo "Data preparation complete!"
}

# ======================== EXPERIMENTS ========================

# Experiment 1: Baseline — Original TinyZero (binary reward)
run_baseline() {
    local model_name=$1  # e.g., "0.5b", "1.5b", "3b"
    local model_path=$2
    local n_gpus=$3
    
    echo "========================================"
    echo "Baseline: Qwen2.5-${model_name} + Binary Reward"
    echo "========================================"
    
    export N_GPUS=$n_gpus
    export BASE_MODEL=$model_path
    export ROLLOUT_TP_SIZE=$n_gpus
    export EXPERIMENT_NAME="baseline-${model_name}"
    
    python -m verl.trainer.main_ppo \
        data.train_files=$DATA_DIR/original/train.parquet \
        data.val_files=$DATA_DIR/original/test.parquet \
        data.train_batch_size=256 \
        data.val_batch_size=1024 \
        data.max_prompt_length=256 \
        data.max_response_length=1024 \
        actor_rollout_ref.model.path=$BASE_MODEL \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=128 \
        actor_rollout_ref.actor.ppo_micro_batch_size=8 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
        actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
        critic.optim.lr=1e-5 \
        critic.model.path=$BASE_MODEL \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.logger='[wandb]' \
        trainer.project_name=NSM-TinyZero \
        trainer.experiment_name=$EXPERIMENT_NAME \
        trainer.n_gpus_per_node=$N_GPUS \
        trainer.nnodes=1 \
        trainer.save_freq=50 \
        trainer.test_freq=50 \
        trainer.total_epochs=15 \
        trainer.default_hdfs_dir=null \
        +trainer.val_before_train=True
}

# Experiment 2: NSM Reward (our method)
run_nsm() {
    local model_name=$1
    local model_path=$2
    local n_gpus=$3
    
    echo "========================================"
    echo "NSM: Qwen2.5-${model_name} + NSM Reward"
    echo "========================================"
    
    export N_GPUS=$n_gpus
    export BASE_MODEL=$model_path
    export ROLLOUT_TP_SIZE=$n_gpus
    export EXPERIMENT_NAME="nsm-${model_name}"
    
    python -m verl.trainer.main_ppo \
        data.train_files=$DATA_DIR/original/train.parquet \
        data.val_files=$DATA_DIR/original/test.parquet \
        data.train_batch_size=256 \
        data.val_batch_size=1024 \
        data.max_prompt_length=256 \
        data.max_response_length=1024 \
        actor_rollout_ref.model.path=$BASE_MODEL \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=128 \
        actor_rollout_ref.actor.ppo_micro_batch_size=8 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
        actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
        critic.optim.lr=1e-5 \
        critic.model.path=$BASE_MODEL \
        algorithm.kl_ctrl.kl_coef=0.001 \
        reward.custom_reward_function.path=$NSM_REWARD_PATH \
        reward.custom_reward_function.name=compute_score \
        trainer.logger='[wandb]' \
        trainer.project_name=NSM-TinyZero \
        trainer.experiment_name=$EXPERIMENT_NAME \
        trainer.n_gpus_per_node=$N_GPUS \
        trainer.nnodes=1 \
        trainer.save_freq=50 \
        trainer.test_freq=50 \
        trainer.total_epochs=15 \
        trainer.default_hdfs_dir=null \
        +trainer.val_before_train=True
}

# Experiment 3: NSM + Warm-start SFT
run_nsm_warmstart() {
    local model_name=$1
    local model_path=$2
    local n_gpus=$3
    local warmstart_samples=${4:-100}
    
    echo "========================================"
    echo "NSM+WarmStart: Qwen2.5-${model_name}"
    echo "  Warm-start samples: ${warmstart_samples}"
    echo "========================================"
    
    # Step 1: Warm-start SFT
    local warmstart_ckpt="$CKPT_DIR/warmstart-${model_name}-n${warmstart_samples}"
    
    if [ ! -d "$warmstart_ckpt" ]; then
        echo "Running warm-start SFT..."
        python $PROJ_DIR/scripts/warmstart_sft.py \
            --model_path $model_path \
            --data_path $DATA_DIR/multidiff/warmstart.parquet \
            --output_dir $warmstart_ckpt \
            --n_samples $warmstart_samples \
            --epochs 2 \
            --lr 5e-6 \
            --batch_size 8
    else
        echo "Warm-start checkpoint exists, skipping SFT."
    fi
    
    # Step 2: RL with NSM reward on warm-started model
    export N_GPUS=$n_gpus
    export BASE_MODEL=$warmstart_ckpt
    export ROLLOUT_TP_SIZE=$n_gpus
    export EXPERIMENT_NAME="nsm-warmstart-${model_name}-n${warmstart_samples}"
    
    python -m verl.trainer.main_ppo \
        data.train_files=$DATA_DIR/original/train.parquet \
        data.val_files=$DATA_DIR/original/test.parquet \
        data.train_batch_size=256 \
        data.val_batch_size=1024 \
        data.max_prompt_length=256 \
        data.max_response_length=1024 \
        actor_rollout_ref.model.path=$BASE_MODEL \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=128 \
        actor_rollout_ref.actor.ppo_micro_batch_size=8 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
        actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
        critic.optim.lr=1e-5 \
        critic.model.path=$BASE_MODEL \
        algorithm.kl_ctrl.kl_coef=0.001 \
        reward.custom_reward_function.path=$NSM_REWARD_PATH \
        reward.custom_reward_function.name=compute_score \
        trainer.logger='[wandb]' \
        trainer.project_name=NSM-TinyZero \
        trainer.experiment_name=$EXPERIMENT_NAME \
        trainer.n_gpus_per_node=$N_GPUS \
        trainer.nnodes=1 \
        trainer.save_freq=50 \
        trainer.test_freq=50 \
        trainer.total_epochs=15 \
        trainer.default_hdfs_dir=null \
        +trainer.val_before_train=True
}

# ======================== ABLATION EXPERIMENTS ========================

run_ablation_no_proximity() {
    echo "Ablation: NSM without proximity reward"
    # Same as run_nsm but with proximity_weight=0
    # Modify extra_info in the reward function config
    export EXPERIMENT_NAME="ablation-no-proximity-0.5b"
    # ... (same as run_nsm but pass proximity_weight=0)
}

run_ablation_no_reasoning_bonus() {
    echo "Ablation: NSM without reasoning quality bonus"
    export EXPERIMENT_NAME="ablation-no-reasoning-0.5b"
    # ... (same as run_nsm but pass reasoning_bonus_weight=0)
}

run_ablation_warmstart_samples() {
    echo "Ablation: Warm-start with different sample sizes"
    for n in 10 50 100 200 500; do
        run_nsm_warmstart "0.5b" $MODEL_05B 1 $n
    done
}

# ======================== MAIN ========================

case "${1:-all}" in
    prepare)
        prepare_data
        ;;
    baseline_05b)
        run_baseline "0.5b" $MODEL_05B 1
        ;;
    baseline_15b)
        run_baseline "1.5b" $MODEL_15B 2
        ;;
    baseline_3b)
        run_baseline "3b" $MODEL_3B 2
        ;;
    nsm_05b)
        run_nsm "0.5b" $MODEL_05B 1
        ;;
    nsm_15b)
        run_nsm "1.5b" $MODEL_15B 2
        ;;
    nsm_3b)
        run_nsm "3b" $MODEL_3B 2
        ;;
    warmstart_05b)
        run_nsm_warmstart "0.5b" $MODEL_05B 1
        ;;
    warmstart_15b)
        run_nsm_warmstart "1.5b" $MODEL_15B 2
        ;;
    ablation_samples)
        run_ablation_warmstart_samples
        ;;
    all)
        prepare_data
        
        echo ""
        echo "============================================"
        echo "Running all experiments (this will take a while)"
        echo "============================================"
        echo ""
        
        # 0.5B experiments (the key experiment!)
        run_baseline "0.5b" $MODEL_05B 1
        run_nsm "0.5b" $MODEL_05B 1
        run_nsm_warmstart "0.5b" $MODEL_05B 1
        
        # 1.5B experiments
        run_baseline "1.5b" $MODEL_15B 2
        run_nsm "1.5b" $MODEL_15B 2
        
        # 3B experiments
        run_baseline "3b" $MODEL_3B 2
        run_nsm "3b" $MODEL_3B 2
        
        echo ""
        echo "All experiments complete!"
        echo "Check wandb project 'NSM-TinyZero' for results."
        ;;
    *)
        echo "Usage: bash run_experiments.sh [experiment]"
        echo ""
        echo "Experiments:"
        echo "  prepare            - Generate datasets"
        echo "  baseline_05b/15b/3b - Baseline (original binary reward)"
        echo "  nsm_05b/15b/3b     - NSM reward (our method)"
        echo "  warmstart_05b/15b  - NSM + Warm-start"
        echo "  ablation_samples   - Warm-start sample size ablation"
        echo "  all                - Run everything"
        ;;
esac
