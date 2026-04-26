"""
Analysis & Visualization for NSM Paper Figures
================================================
Generates key figures for the paper:
1. Learning curves (accuracy over training steps)
2. Error type distribution over training
3. Reward distribution comparison (binary vs NSM)
4. Warm-start ablation (accuracy vs n_samples)
5. Aha moment detection (response length & self-verification emergence)

Usage:
    python analyze_results.py --wandb_project NSM-TinyZero --output_dir ./figures
    
    # Or from local logs:
    python analyze_results.py --log_dir ./logs --output_dir ./figures
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict, Counter

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.linewidth'] = 1.2


# ============================================================
# Color scheme for the paper
# ============================================================
COLORS = {
    'baseline': '#999999',
    'nsm': '#E8553A',
    'nsm_warmstart': '#2D7DD2',
    'nsm_annealed': '#45B7A0',
    
    # Error types
    'correct': '#2ECC71',
    'arithmetic': '#F39C12',
    'partial': '#E67E22',
    'number_usage': '#E74C3C',
    'format': '#9B59B6',
    'no_answer': '#95A5A6',
    'empty': '#BDC3C7',
}


# ============================================================
# Figure 1: Learning Curves (Main Result)
# ============================================================
def plot_learning_curves(results: dict, output_path: str):
    """
    Plot accuracy over training steps for different methods.
    This is the most important figure — shows NSM enables 0.5B to learn.
    
    Args:
        results: dict of {method_name: {'steps': [...], 'accuracy': [...]}}
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    model_sizes = ['0.5B', '1.5B', '3B']
    
    for ax, size in zip(axes, model_sizes):
        for method_name, data in results.items():
            if size.lower() not in method_name.lower():
                continue
            
            color_key = 'baseline'
            if 'nsm' in method_name.lower() and 'warmstart' in method_name.lower():
                color_key = 'nsm_warmstart'
            elif 'nsm' in method_name.lower():
                color_key = 'nsm'
            
            label = method_name.split('-')[0]
            ax.plot(
                data['steps'], data['accuracy'],
                color=COLORS[color_key],
                linewidth=2.5,
                label=label,
                alpha=0.9,
            )
        
        ax.set_title(f'Qwen2.5-{size}', fontweight='bold', fontsize=13)
        ax.set_xlabel('Training Steps')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        
        if size == '0.5B':
            ax.set_ylabel('Accuracy')
            # Add annotation for the key result
            ax.annotate(
                'Binary reward: stuck at 0%',
                xy=(0.5, 0.02), xycoords='axes fraction',
                fontsize=9, color=COLORS['baseline'],
                fontstyle='italic',
            )
    
    axes[0].legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    fig.suptitle(
        'Negative Sample Mining Enables Small Model Zero-RL',
        fontsize=14, fontweight='bold', y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================
# Figure 2: Error Type Distribution Over Training
# ============================================================
def plot_error_distribution(error_data: dict, output_path: str):
    """
    Stacked area chart showing how error types shift during training.
    Shows the model gradually reducing "empty" and "no_answer" errors.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    error_types = [
        'correct', 'arithmetic', 'partial',
        'number_usage', 'format', 'no_answer', 'empty'
    ]
    
    for ax, (method_name, data) in zip(axes, error_data.items()):
        steps = data['steps']
        
        # Stack the error type fractions
        bottoms = np.zeros(len(steps))
        for etype in error_types:
            values = np.array(data.get(etype, [0]*len(steps)))
            ax.fill_between(
                steps, bottoms, bottoms + values,
                color=COLORS.get(etype, '#ccc'),
                label=etype,
                alpha=0.8,
            )
            bottoms += values
        
        ax.set_title(method_name, fontweight='bold')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Fraction')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.2)
    
    axes[0].legend(
        loc='upper right', fontsize=8, ncol=2, framealpha=0.9
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================
# Figure 3: Reward Distribution Comparison
# ============================================================
def plot_reward_distributions(
    binary_rewards: list,
    nsm_rewards: list,
    output_path: str,
):
    """
    Histogram comparing reward distributions.
    Binary: two spikes at 0 and 1 (mostly 0).
    NSM: spread across (0, 1) with meaningful gradation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    axes[0].hist(
        binary_rewards, bins=50, color=COLORS['baseline'],
        alpha=0.8, edgecolor='white',
    )
    axes[0].set_title('Binary Reward (Baseline)', fontweight='bold')
    axes[0].set_xlabel('Reward')
    axes[0].set_ylabel('Count')
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.3)
    
    axes[1].hist(
        nsm_rewards, bins=50, color=COLORS['nsm'],
        alpha=0.8, edgecolor='white',
    )
    axes[1].set_title('NSM Reward (Ours)', fontweight='bold')
    axes[1].set_xlabel('Reward')
    
    # Annotate
    binary_zero_frac = sum(1 for r in binary_rewards if r < 0.01) / len(binary_rewards)
    nsm_zero_frac = sum(1 for r in nsm_rewards if r < 0.01) / len(nsm_rewards)
    
    axes[0].annotate(
        f'{binary_zero_frac:.0%} zero reward',
        xy=(0.05, 0.9), xycoords='axes fraction',
        fontsize=10, color='red', fontweight='bold',
    )
    axes[1].annotate(
        f'{nsm_zero_frac:.0%} zero reward',
        xy=(0.05, 0.9), xycoords='axes fraction',
        fontsize=10, color='green', fontweight='bold',
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================
# Figure 4: Warm-start Ablation
# ============================================================
def plot_warmstart_ablation(
    sample_sizes: list,
    final_accuracies: list,
    output_path: str,
):
    """
    Bar chart: final accuracy vs number of warm-start SFT samples.
    Shows that even 50-100 samples are sufficient.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    bars = ax.bar(
        range(len(sample_sizes)),
        final_accuracies,
        color=[COLORS['nsm_warmstart'] if s > 0 else COLORS['baseline']
               for s in sample_sizes],
        edgecolor='white',
        linewidth=1.5,
    )
    
    ax.set_xticks(range(len(sample_sizes)))
    ax.set_xticklabels(
        [str(s) if s > 0 else '0\n(no warmstart)' for s in sample_sizes],
        fontsize=10,
    )
    ax.set_xlabel('Warm-start SFT Samples', fontsize=12)
    ax.set_ylabel('Final Accuracy', fontsize=12)
    ax.set_title(
        'Warm-start Ablation (Qwen2.5-0.5B)',
        fontweight='bold', fontsize=13,
    )
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, acc in zip(bars, final_accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f'{acc:.1%}',
            ha='center', fontsize=10, fontweight='bold',
        )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================
# Figure 5: Aha Moment Analysis
# ============================================================
def plot_aha_moment(
    results: dict,
    output_path: str,
):
    """
    Dual-axis plot: response length + self-verification rate over training.
    Shows when the "Aha moment" (self-verification emergence) happens.
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    
    for method_name, data in results.items():
        color_key = 'nsm' if 'nsm' in method_name.lower() else 'baseline'
        
        ax1.plot(
            data['steps'], data['avg_response_length'],
            color=COLORS[color_key],
            linewidth=2, label=f'{method_name} (length)',
            linestyle='-',
        )
        
        ax2.plot(
            data['steps'], data['self_verification_rate'],
            color=COLORS[color_key],
            linewidth=2, label=f'{method_name} (verification)',
            linestyle='--', alpha=0.7,
        )
    
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Avg Response Length (tokens)', color='black')
    ax2.set_ylabel('Self-Verification Rate', color='gray')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    
    ax1.set_title(
        'Reasoning Behavior Emergence ("Aha Moment")',
        fontweight='bold', fontsize=13,
    )
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================
# Offline Reward Analysis (no training needed)
# ============================================================
def analyze_reward_offline(
    model_outputs_path: str,
    ground_truth_path: str,
    output_dir: str,
):
    """
    Analyze reward distributions on a set of model outputs.
    Useful for comparing binary vs NSM rewards without retraining.
    
    This lets you:
    1. Take existing TinyZero checkpoints
    2. Generate rollouts
    3. Score them with both reward functions
    4. Compare distributions
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from reward_functions.countdown_nsm import (
        compute_score, compute_score_with_details, _compute_binary_score
    )
    
    # Load data
    outputs = json.load(open(model_outputs_path))
    
    binary_rewards = []
    nsm_rewards = []
    details_list = []
    
    for item in outputs:
        solution_str = item['response']
        ground_truth = item['ground_truth']
        
        binary = _compute_binary_score(
            solution_str, ground_truth['target'], ground_truth['numbers']
        )
        
        detail = compute_score_with_details(
            data_source='countdown',
            solution_str=solution_str,
            ground_truth=ground_truth,
        )
        
        binary_rewards.append(binary)
        nsm_rewards.append(detail['score'])
        details_list.append(detail)
    
    # Plot reward distributions
    plot_reward_distributions(
        binary_rewards, nsm_rewards,
        os.path.join(output_dir, 'fig3_reward_distributions.pdf'),
    )
    
    # Print statistics
    print("\n" + "=" * 50)
    print("Reward Distribution Analysis")
    print("=" * 50)
    print(f"Total samples: {len(binary_rewards)}")
    print(f"\nBinary Reward:")
    print(f"  Mean: {np.mean(binary_rewards):.4f}")
    print(f"  Zero-reward fraction: {sum(1 for r in binary_rewards if r < 0.01) / len(binary_rewards):.1%}")
    print(f"  Correct fraction: {sum(1 for r in binary_rewards if r > 0.99) / len(binary_rewards):.1%}")
    
    print(f"\nNSM Reward:")
    print(f"  Mean: {np.mean(nsm_rewards):.4f}")
    print(f"  Zero-reward fraction: {sum(1 for r in nsm_rewards if r < 0.01) / len(nsm_rewards):.1%}")
    print(f"  Partial-reward fraction: {sum(1 for r in nsm_rewards if 0.01 < r < 0.99) / len(nsm_rewards):.1%}")
    
    # Error type breakdown
    error_counter = Counter(d['error_type'] for d in details_list)
    print(f"\nError Type Distribution:")
    for etype, count in error_counter.most_common():
        print(f"  {etype:20s}: {count:5d} ({count/len(details_list):.1%})")


# ============================================================
# Unit Test for Reward Function
# ============================================================
def test_reward_function():
    """
    Quick unit test to verify the reward function works correctly.
    Run this before starting experiments!
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from reward_functions.countdown_nsm import (
        compute_score, compute_score_with_details
    )
    
    test_cases = [
        # (description, solution_str, ground_truth, expected_range)
        (
            "Correct answer",
            "<think>Let me try: 5 + 3 = 8</think><answer>5 + 3</answer>",
            {'target': 8, 'numbers': [5, 3]},
            (1.0, 1.0),
        ),
        (
            "Wrong computation, right numbers",
            "<think>5 * 3 = 20? No wait...</think><answer>5 * 3</answer>",
            {'target': 8, 'numbers': [5, 3]},
            (0.4, 0.85),  # arithmetic error + full number overlap + some proximity
        ),
        (
            "Wrong numbers used",
            "<think>Let me try 4 + 4</think><answer>4 + 4</answer>",
            {'target': 8, 'numbers': [5, 3]},
            (0.2, 0.65),  # number_usage error but result equals target (proximity=1)
        ),
        (
            "Format error (unparseable)",
            "<think>hmm</think><answer>five plus three</answer>",
            {'target': 8, 'numbers': [5, 3]},
            (0.05, 0.2),
        ),
        (
            "No answer tags",
            "I think the answer is 8",
            {'target': 8, 'numbers': [5, 3]},
            (0.0, 0.1),
        ),
        (
            "Empty response",
            "",
            {'target': 8, 'numbers': [5, 3]},
            (0.0, 0.01),
        ),
        (
            "Close but wrong (proximity test)",
            "<think>44 + 59 = 103, 103 - 22 = 81, close to 37...</think><answer>44 - 22 + 59</answer>",
            {'target': 37, 'numbers': [44, 59, 22]},
            (0.4, 0.85),  # arithmetic error (correct numbers, all used) + reasoning steps
        ),
        (
            "Self-verification present",
            "<think>Let me try: 59 - 22 = 37. Let me check: 59 - 22 = 37. Yes!</think><answer>59 - 22</answer>",
            {'target': 37, 'numbers': [44, 59, 22]},
            (0.3, 0.85),  # number_usage (missing 44) but high proximity + verification bonus
        ),
    ]
    
    print("=" * 60)
    print("Unit Tests for NSM Reward Function")
    print("=" * 60)
    
    all_passed = True
    for desc, sol, gt, (exp_min, exp_max) in test_cases:
        detail = compute_score_with_details('countdown', sol, gt)
        score = detail['score']
        passed = exp_min <= score <= exp_max
        
        status = "✓" if passed else "✗"
        print(f"\n{status} {desc}")
        print(f"  Score: {score:.4f} (expected [{exp_min}, {exp_max}])")
        print(f"  Error type: {detail['error_type']}")
        print(f"  Proximity: {detail['proximity']:.3f}, "
              f"Overlap: {detail['number_overlap']:.3f}")
        
        if not passed:
            all_passed = False
            print(f"  *** FAILED ***")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests FAILED — check the reward function.")
    print("=" * 60)
    
    return all_passed


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='test',
                        choices=['test', 'analyze', 'plot_demo'])
    parser.add_argument('--output_dir', type=str, default='./figures')
    parser.add_argument('--model_outputs', type=str, default=None)
    parser.add_argument('--ground_truth', type=str, default=None)
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.action == 'test':
        test_reward_function()
    
    elif args.action == 'analyze':
        if args.model_outputs and args.ground_truth:
            analyze_reward_offline(
                args.model_outputs, args.ground_truth, args.output_dir
            )
        else:
            print("Please provide --model_outputs and --ground_truth")
    
    elif args.action == 'plot_demo':
        # Generate demo figures with synthetic data (for testing the plotting code)
        print("Generating demo figures with synthetic data...")
        
        # Demo: reward distributions
        np.random.seed(42)
        binary = np.random.choice([0.0, 0.1, 1.0], size=1000, p=[0.7, 0.2, 0.1])
        nsm = np.random.beta(2, 5, size=1000) * 0.8
        nsm[np.random.random(1000) < 0.1] = 1.0  # 10% correct
        
        plot_reward_distributions(
            binary.tolist(), nsm.tolist(),
            os.path.join(args.output_dir, 'demo_fig3_reward_dist.pdf'),
        )
        
        # Demo: warmstart ablation
        plot_warmstart_ablation(
            sample_sizes=[0, 10, 50, 100, 200, 500],
            final_accuracies=[0.0, 0.05, 0.28, 0.35, 0.37, 0.36],
            output_path=os.path.join(args.output_dir, 'demo_fig4_warmstart.pdf'),
        )
        
        print(f"Demo figures saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
