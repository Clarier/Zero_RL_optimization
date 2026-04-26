# Negative Sample Mining for Zero-RL (NSM-TinyZero)

> **Enabling Small Language Models to Learn Reasoning via Zero-RL**
>
> Addresses TinyZero's known limitation: Qwen2.5-0.5B fails to learn reasoning with standard binary rewards. Our method provides fine-grained learning signals even when all samples are incorrect.

## Core Idea

Standard zero-RL (like DeepSeek R1-Zero / TinyZero) uses binary rewards:
- Correct answer → 1.0
- Wrong answer → 0.0 (or 0.1 for format)

**Problem:** Weak models (≤1B params) produce *all wrong answers* initially → all rewards are 0 → GRPO advantage is 0 → model learns nothing.

**Our Solution — Negative Sample Mining (NSM):**

1. **Error-type-aware scoring**: Distinguish *how* the model failed (format error vs. arithmetic error vs. wrong numbers) and assign graduated partial rewards
2. **Proximity-based reward**: "How close" is the computed result to the target
3. **Reasoning quality bonus**: Reward models that show reasoning steps, even if the final answer is wrong
4. **Warm-start bootstrap**: Optional 1-2 epoch SFT on ~100 easy problems to kick-start RL

## Project Structure

```
negative_sample_mining/
├── README.md                              # This file
├── reward_functions/
│   ├── countdown_nsm.py                   # ★ Core: NSM reward function (drop-in replacement)
│   └── main_ppo_nsm.py                    # Enhanced RewardManager with logging
├── data_preprocess/
│   └── countdown_multidiff.py             # Multi-difficulty dataset generator
├── scripts/
│   ├── warmstart_sft.py                   # Lightweight warm-start SFT
│   └── run_experiments.sh                 # Full experiment pipeline
└── analysis/
    └── analyze_results.py                 # Paper figures & unit tests
```

## Quick Start

### 1. Verify the reward function works

```bash
cd negative_sample_mining
python analysis/analyze_results.py --action test
```

### 2. Integrate into TinyZero

**Option A — Custom reward function (recommended, zero code change):**
```bash
# Just point veRL to our reward function
python -m verl.trainer.main_ppo \
    ... \
    reward.custom_reward_function.path=reward_functions/countdown_nsm.py \
    reward.custom_reward_function.name=compute_score
```

**Option B — Replace the original file:**
```bash
cp reward_functions/countdown_nsm.py TinyZero/verl/utils/reward_score/countdown.py
```

### 3. Run experiments

```bash
# Prepare datasets
bash scripts/run_experiments.sh prepare

# Run the key experiment: 0.5B model (should fail with baseline, succeed with NSM)
bash scripts/run_experiments.sh baseline_05b   # Baseline: expect ~0% accuracy
bash scripts/run_experiments.sh nsm_05b        # NSM: expect improvement
bash scripts/run_experiments.sh warmstart_05b  # NSM+Warmstart: expect best

# Run all experiments
bash scripts/run_experiments.sh all
```

## Experiment Plan for the Paper

### Main Experiments (Table 1)

| Model | Method | Expected Accuracy | GPU-hours |
|-------|--------|-------------------|-----------|
| Qwen2.5-0.5B | Baseline (Binary) | ~0% | 2h × 1GPU |
| Qwen2.5-0.5B | NSM | ~15-25% | 2h × 1GPU |
| Qwen2.5-0.5B | NSM + WarmStart | ~30-40% | 2.5h × 1GPU |
| Qwen2.5-1.5B | Baseline | ~40% | 4h × 2GPU |
| Qwen2.5-1.5B | NSM | ~50-60% | 4h × 2GPU |
| Qwen2.5-3B | Baseline | ~70% | 6h × 2GPU |
| Qwen2.5-3B | NSM | ~75-80% | 6h × 2GPU |

### Ablation Experiments (Table 2)

| Ablation | What's removed | Expected effect |
|----------|---------------|-----------------|
| No proximity reward | `proximity_weight=0` | Moderate drop |
| No reasoning bonus | `reasoning_bonus_weight=0` | Small drop |
| No number overlap | `number_overlap_weight=0` | Moderate drop |
| Binary + WarmStart only | `reward_mode='binary'` + warmstart | Partial improvement |

### Warm-start Sample Size (Figure 4)

Test n = {0, 10, 50, 100, 200, 500} warm-start samples for 0.5B model.
Expected: plateau around n=100, diminishing returns after.

### Key Figures

1. **Figure 1**: Learning curves — 3 panels (0.5B, 1.5B, 3B), each with Baseline/NSM/NSM+WS
2. **Figure 2**: Error type distribution shift over training
3. **Figure 3**: Reward distribution comparison (binary vs NSM)
4. **Figure 4**: Warm-start ablation
5. **Figure 5**: Aha moment (response length + self-verification emergence)

## NSM Reward Function Details

The reward function scores each response on a scale of [0, 1]:

```
Score = ErrorBaseline + Proximity × w₁ + NumberOverlap × w₂ + ReasoningBonus × w₃

ErrorBaseline:
  correct       → 1.0 (returned directly)
  arithmetic    → 0.4  (right numbers, wrong computation)
  partial       → 0.3  (correct subset computation)
  number_usage  → 0.2  (wrong numbers)
  format        → 0.1  (has <answer> but unparseable)
  no_answer     → 0.05 (missing <answer> tags)
  empty         → 0.0

Proximity:  1 - |result - target| / |target|     (capped at [0,1])
NumberOverlap: |used ∩ expected| / |expected|
ReasoningBonus: min(n_steps/5, 1) × w₃ + has_verification × 0.05
```

## Total Compute Budget

Estimated for 4× A100-80G:
- Data preparation: ~10 min
- 0.5B experiments (3 methods): ~6 GPU-hours
- 1.5B experiments (2 methods): ~16 GPU-hours  
- 3B experiments (2 methods): ~24 GPU-hours
- Ablations: ~12 GPU-hours
- **Total: ~58 GPU-hours ≈ 3 days on 4× A100**

## Citation

If you use this code, please cite both TinyZero and this work:

```bibtex
@misc{nsm_tinyzero,
    title={Negative Sample Mining for Zero-RL: Enabling Small Model Reasoning},
    year={2026},
}

@misc{tinyzero,
    author={Jiayi Pan and Junjie Zhang and Xingyao Wang and Lifan Yuan and Hao Peng and Alane Suhr},
    title={TinyZero},
    howpublished={https://github.com/Jiayi-Pan/TinyZero},
    year={2025},
}
```
