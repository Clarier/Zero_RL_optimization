"""
Multi-Difficulty Countdown Dataset Preprocessor
================================================
Extended from TinyZero's examples/data_preprocess/countdown.py

Generates countdown tasks with controlled difficulty levels:
- Easy:   2 numbers, small target (1-20)
- Medium: 3 numbers, medium target (1-50)  [original TinyZero]
- Hard:   4 numbers, larger target (1-100)
- Expert: 5 numbers, large target, may require division

This supports:
1. Warm-start data generation (easy subset for initial SFT)
2. Curriculum learning experiments
3. Difficulty-stratified evaluation
"""

import os
import json
import random
import argparse
import itertools
from typing import List, Tuple, Optional
from collections import defaultdict

import pandas as pd


# ============================================================
# Countdown Problem Generator
# ============================================================
def generate_countdown_problem(
    n_numbers: int = 4,
    max_number: int = 99,
    target_range: Tuple[int, int] = (1, 100),
    require_solvable: bool = True,
    max_attempts: int = 100,
) -> Optional[dict]:
    """
    Generate a single countdown problem.
    
    Args:
        n_numbers: How many numbers to provide
        max_number: Maximum value for any individual number
        target_range: (min, max) for the target value
        require_solvable: If True, only return problems with known solutions
        max_attempts: Max attempts to find a solvable problem
    
    Returns:
        dict with 'numbers', 'target', 'difficulty', 'has_solution'
    """
    for _ in range(max_attempts):
        numbers = [random.randint(1, max_number) for _ in range(n_numbers)]
        target = random.randint(target_range[0], target_range[1])
        
        if require_solvable:
            if _is_solvable(numbers, target):
                return {
                    'numbers': numbers,
                    'target': target,
                    'n_numbers': n_numbers,
                    'has_solution': True,
                }
        else:
            return {
                'numbers': numbers,
                'target': target,
                'n_numbers': n_numbers,
                'has_solution': _is_solvable(numbers, target),
            }
    
    # Fallback: just return it without guarantee
    return {
        'numbers': numbers,
        'target': target,
        'n_numbers': n_numbers,
        'has_solution': False,
    }


def _is_solvable(numbers: List[int], target: int) -> bool:
    """
    Check if target can be reached using the given numbers with +,-,*,/.
    Uses recursive search over all permutations and operations.
    Limited to small n for efficiency.
    """
    if len(numbers) > 6:
        return True  # Skip check for large sets (too expensive)
    
    return _search_solution(numbers, target)


def _search_solution(nums: List[int], target: int) -> bool:
    """Recursive search for a valid equation."""
    if len(nums) == 1:
        return abs(nums[0] - target) < 1e-9
    
    for i in range(len(nums)):
        for j in range(len(nums)):
            if i == j:
                continue
            remaining = [nums[k] for k in range(len(nums)) if k != i and k != j]
            
            a, b = nums[i], nums[j]
            
            # Try all operations
            candidates = [a + b, a - b, a * b]
            if b != 0:
                candidates.append(a / b)
            
            for result in candidates:
                if _search_solution(remaining + [result], target):
                    return True
    
    return False


# ============================================================
# Difficulty Levels
# ============================================================
DIFFICULTY_CONFIGS = {
    'easy': {
        'n_numbers': 2,
        'max_number': 20,
        'target_range': (1, 20),
        'description': '2 numbers, small targets',
    },
    'medium': {
        'n_numbers': 3,
        'max_number': 50,
        'target_range': (1, 50),
        'description': '3 numbers, medium targets (similar to original TinyZero)',
    },
    'hard': {
        'n_numbers': 4,
        'max_number': 99,
        'target_range': (1, 100),
        'description': '4 numbers, larger targets',
    },
    'expert': {
        'n_numbers': 5,
        'max_number': 99,
        'target_range': (10, 200),
        'description': '5 numbers, large targets, may need division',
    },
}


# ============================================================
# Prompt Templates
# ============================================================
def make_prefix(numbers, target, template_type='base'):
    """Generate the prompt prefix."""
    numbers_str = str(numbers)
    
    if template_type == 'base':
        prefix = (
            f"A conversation between User and Assistant. The user asks a "
            f"question, and the Assistant solves it. The assistant first "
            f"thinks about the reasoning process in the mind and then "
            f"provides the user with the answer.\n"
            f"User: Using the numbers {numbers_str}, create an equation "
            f"that equals {target}. You can use basic arithmetic operations "
            f"(+, -, *, /) and each number can only be used once. Show your "
            f"work in <think> </think> tags. And return the final answer in "
            f"<answer> </answer> tags, for example <answer> (1 + 2) / 3 "
            f"</answer>.\n"
            f"Assistant: Let me solve this step by step.\n<think>"
        )
    elif template_type == 'qwen-instruct':
        prefix = (
            f"<|im_start|>system\nYou are a helpful assistant. You first "
            f"thinks about the reasoning process in the mind and then "
            f"provides the user with the answer.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Using the numbers {numbers_str}, create an equation that "
            f"equals {target}. You can use basic arithmetic operations "
            f"(+, -, *, /) and each number can only be used once. Show your "
            f"work in <think> </think> tags. And return the final answer in "
            f"<answer> </answer> tags, for example <answer> (1 + 2) / 3 "
            f"</answer>.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    else:
        raise ValueError(f"Unknown template type: {template_type}")
    
    return prefix


# ============================================================
# Dataset Generation
# ============================================================
def generate_dataset(
    difficulties: List[str],
    samples_per_difficulty: int,
    template_type: str = 'base',
    data_source: str = 'countdown',
    require_solvable: bool = True,
    seed: int = 42,
) -> List[dict]:
    """
    Generate a dataset with specified difficulty distribution.
    """
    random.seed(seed)
    dataset = []
    
    for difficulty in difficulties:
        config = DIFFICULTY_CONFIGS[difficulty]
        print(f"Generating {samples_per_difficulty} {difficulty} samples "
              f"({config['description']})...")
        
        for i in range(samples_per_difficulty):
            problem = generate_countdown_problem(
                n_numbers=config['n_numbers'],
                max_number=config['max_number'],
                target_range=config['target_range'],
                require_solvable=require_solvable,
            )
            
            if problem is None:
                continue
            
            prompt = make_prefix(
                problem['numbers'],
                problem['target'],
                template_type=template_type,
            )
            
            solution = {
                'target': problem['target'],
                'numbers': problem['numbers'],
            }
            
            data_item = {
                'data_source': data_source,
                'prompt': [{'role': 'user', 'content': prompt}],
                'ability': 'math',
                'reward_model': {
                    'style': 'rule',
                    'ground_truth': solution,
                },
                'extra_info': {
                    'difficulty': difficulty,
                    'n_numbers': problem['n_numbers'],
                    'has_solution': problem['has_solution'],
                    'split': 'train',
                },
            }
            dataset.append(data_item)
    
    random.shuffle(dataset)
    return dataset


def generate_warmstart_data(
    n_samples: int = 100,
    template_type: str = 'base',
    seed: int = 42,
) -> List[dict]:
    """
    Generate easy problems WITH solutions for warm-start SFT.
    
    These are very easy problems (2 numbers) where we also provide
    a correct solution in the training data. Used for 1-2 steps of
    lightweight SFT before switching to pure RL.
    """
    random.seed(seed)
    dataset = []
    
    for _ in range(n_samples):
        # Generate very easy problems
        a = random.randint(1, 20)
        b = random.randint(1, 20)
        
        # Choose a random operation and compute target
        op = random.choice(['+', '-', '*'])
        if op == '+':
            target = a + b
            solution_expr = f"{a} + {b}"
        elif op == '-':
            target = a - b
            if target <= 0:
                target = a + b
                solution_expr = f"{a} + {b}"
            else:
                solution_expr = f"{a} - {b}"
        else:  # *
            target = a * b
            solution_expr = f"{a} * {b}"
        
        numbers = [a, b]
        
        prompt = make_prefix(numbers, target, template_type=template_type)
        
        # For warm-start, we include the solution
        solution_text = (
            f"\nLet me think about this.\n"
            f"I need to use {numbers} to make {target}.\n"
            f"{solution_expr} = {target}\n"
            f"</think>\n"
            f"<answer>{solution_expr}</answer>"
        )
        
        data_item = {
            'data_source': 'countdown',
            'prompt': [{'role': 'user', 'content': prompt}],
            'response': solution_text,  # Include solution for SFT
            'ability': 'math',
            'reward_model': {
                'style': 'rule',
                'ground_truth': {
                    'target': target,
                    'numbers': numbers,
                },
            },
            'extra_info': {
                'difficulty': 'warmstart',
                'n_numbers': 2,
                'has_solution': True,
                'split': 'train',
            },
        }
        dataset.append(data_item)
    
    return dataset


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Generate multi-difficulty countdown dataset for NSM'
    )
    parser.add_argument(
        '--local_dir', type=str, required=True,
        help='Output directory for parquet files'
    )
    parser.add_argument(
        '--template_type', type=str, default='base',
        choices=['base', 'qwen-instruct'],
    )
    parser.add_argument(
        '--difficulties', nargs='+',
        default=['easy', 'medium', 'hard'],
        choices=list(DIFFICULTY_CONFIGS.keys()),
        help='Difficulty levels to include'
    )
    parser.add_argument(
        '--train_size', type=int, default=100000,
        help='Total training samples (split across difficulties)'
    )
    parser.add_argument(
        '--test_size', type=int, default=1024,
        help='Total test samples (split across difficulties)'
    )
    parser.add_argument(
        '--warmstart_size', type=int, default=100,
        help='Number of warm-start SFT samples (0 to skip)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
    )
    parser.add_argument(
        '--original_mode', action='store_true',
        help='Generate dataset identical to original TinyZero (for baseline)'
    )
    
    args = parser.parse_args()
    os.makedirs(args.local_dir, exist_ok=True)
    
    if args.original_mode:
        # Generate dataset identical to original TinyZero
        # Uses HuggingFace dataset: Jiayi-Pan/Countdown-Tasks-3to4
        print("Original mode: generating 3-4 number countdown (like TinyZero)")
        from datasets import load_dataset
        raw_dataset = load_dataset('Jiayi-Pan/Countdown-Tasks-3to4', split='train')
        
        train_size = min(args.train_size, len(raw_dataset) - args.test_size)
        train_data = raw_dataset.select(range(train_size))
        test_data = raw_dataset.select(
            range(train_size, train_size + args.test_size)
        )
        
        def process_fn(example):
            prompt = make_prefix(
                example['nums'], example['target'], args.template_type
            )
            return {
                'data_source': 'countdown',
                'prompt': json.dumps([{'role': 'user', 'content': prompt}]),
                'ability': 'math',
                'reward_model': json.dumps({
                    'style': 'rule',
                    'ground_truth': {
                        'target': example['target'],
                        'numbers': example['nums'],
                    },
                }),
            }
        
        train_processed = train_data.map(process_fn)
        test_processed = test_data.map(process_fn)
        
        train_processed.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
        test_processed.to_parquet(os.path.join(args.local_dir, 'test.parquet'))
        print(f"Saved {len(train_processed)} train, {len(test_processed)} test")
        return
    
    # --- Multi-difficulty generation ---
    samples_per_diff_train = args.train_size // len(args.difficulties)
    samples_per_diff_test = args.test_size // len(args.difficulties)
    
    print(f"Generating multi-difficulty dataset:")
    print(f"  Difficulties: {args.difficulties}")
    print(f"  Samples per difficulty (train): {samples_per_diff_train}")
    print(f"  Samples per difficulty (test): {samples_per_diff_test}")
    
    # Training set
    train_data = generate_dataset(
        difficulties=args.difficulties,
        samples_per_difficulty=samples_per_diff_train,
        template_type=args.template_type,
        seed=args.seed,
    )
    
    # Test set
    test_data = generate_dataset(
        difficulties=args.difficulties,
        samples_per_difficulty=samples_per_diff_test,
        template_type=args.template_type,
        seed=args.seed + 1,
    )
    
    # Convert to DataFrames
    def flatten_for_parquet(data_list):
        rows = []
        for item in data_list:
            row = {
                'data_source': item['data_source'],
                'prompt': json.dumps(item['prompt']),
                'ability': item['ability'],
                'reward_model': json.dumps(item['reward_model']),
                'extra_info': json.dumps(item.get('extra_info', {})),
            }
            if 'response' in item:
                row['response'] = item['response']
            rows.append(row)
        return pd.DataFrame(rows)
    
    train_df = flatten_for_parquet(train_data)
    test_df = flatten_for_parquet(test_data)
    
    train_df.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
    test_df.to_parquet(os.path.join(args.local_dir, 'test.parquet'))
    
    print(f"Saved {len(train_df)} train samples to {args.local_dir}/train.parquet")
    print(f"Saved {len(test_df)} test samples to {args.local_dir}/test.parquet")
    
    # Difficulty distribution
    for diff in args.difficulties:
        count = sum(1 for d in train_data 
                    if d.get('extra_info', {}).get('difficulty') == diff)
        print(f"  {diff}: {count} samples")
    
    # Warm-start data
    if args.warmstart_size > 0:
        warmstart_data = generate_warmstart_data(
            n_samples=args.warmstart_size,
            template_type=args.template_type,
            seed=args.seed + 2,
        )
        warmstart_df = flatten_for_parquet(warmstart_data)
        warmstart_df.to_parquet(
            os.path.join(args.local_dir, 'warmstart.parquet')
        )
        print(f"Saved {len(warmstart_df)} warm-start samples to "
              f"{args.local_dir}/warmstart.parquet")


if __name__ == '__main__':
    main()
