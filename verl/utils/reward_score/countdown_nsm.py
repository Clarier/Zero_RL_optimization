"""
Negative Sample Mining (NSM) Reward Function for Countdown Task
================================================================
Drop-in replacement for TinyZero's verl/utils/reward_score/countdown.py

Key innovations over the original binary reward:
1. Error-type-aware scoring: distinguishes format/number/arithmetic/logic errors
2. Proximity-based partial reward: "how close" the answer is to the target
3. Contrastive negative ranking: even among all-wrong samples, rank by quality
4. Warm-start compatible: works with optional lightweight SFT warm-up

Original TinyZero reward: correct=1.0, format_ok=0.1, else=0.0
Our NSM reward: fine-grained scores in [0.0, 1.0] based on error analysis

Usage:
  - Replace verl/utils/reward_score/countdown.py with this file
  - Or specify via: reward.custom_reward_function.path=<path_to_this_file>
"""

import re
import ast
import math
from typing import Optional


# ============================================================
# Error Type Definitions
# ============================================================
class ErrorType:
    """Enumeration of failure modes, ordered by severity (least to most)."""
    CORRECT = "correct"                # Perfect answer
    ARITHMETIC_ERROR = "arithmetic"    # Right numbers, wrong computation
    NUMBER_VIOLATION = "number_usage"  # Used wrong numbers or repeated
    PARTIAL_SOLUTION = "partial"       # Used subset of numbers correctly
    FORMAT_ERROR = "format"            # Has <answer> tags but unparseable
    NO_ANSWER = "no_answer"            # Missing <answer> tags entirely
    EMPTY = "empty"                    # Empty or garbage response

# Severity scores: higher = less severe error = higher partial reward
ERROR_SEVERITY = {
    ErrorType.CORRECT:          1.0,
    ErrorType.ARITHMETIC_ERROR: 0.4,
    ErrorType.PARTIAL_SOLUTION: 0.3,
    ErrorType.NUMBER_VIOLATION: 0.2,
    ErrorType.FORMAT_ERROR:     0.1,
    ErrorType.NO_ANSWER:        0.05,
    ErrorType.EMPTY:            0.0,
}


# ============================================================
# Solution Parsing
# ============================================================
def extract_answer(solution_str: str) -> Optional[str]:
    """Extract content from <answer>...</answer> tags."""
    # Try to find the last <answer> block (model might have multiple attempts)
    matches = re.findall(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def extract_numbers_from_expr(expr: str) -> list:
    """Extract all numeric literals from a math expression string."""
    # Match integers and decimals
    numbers = re.findall(r'\b(\d+\.?\d*)\b', expr)
    result = []
    for n in numbers:
        try:
            val = float(n)
            if val == int(val):
                result.append(int(val))
            else:
                result.append(val)
        except ValueError:
            continue
    return result


def safe_eval_expr(expr: str) -> Optional[float]:
    """
    Safely evaluate a mathematical expression.
    Only allows +, -, *, /, parentheses, and numbers.
    """
    # Sanitize: only allow digits, operators, parens, dots, spaces
    cleaned = re.sub(r'[^0-9+\-*/().  ]', '', expr)
    if not cleaned.strip():
        return None

    try:
        # Use ast for safe evaluation
        tree = ast.parse(cleaned, mode='eval')
        # Verify only safe nodes
        for node in ast.walk(tree):
            if isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp,
                                 ast.Constant, ast.Num,
                                 ast.Add, ast.Sub, ast.Mult, ast.Div,
                                 ast.USub, ast.UAdd)):
                continue
            else:
                return None
        result = eval(compile(tree, '<string>', 'eval'))
        return float(result)
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError,
            OverflowError, RecursionError):
        return None


# ============================================================
# Reasoning Chain Analysis
# ============================================================
def extract_think_content(solution_str: str) -> str:
    """Extract content from <think>...</think> tags."""
    matches = re.findall(r'<think>(.*?)</think>', solution_str, re.DOTALL)
    if matches:
        return matches[-1].strip()
    # If no closing tag, take everything after <think>
    match = re.search(r'<think>(.*)', solution_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def count_reasoning_steps(think_content: str) -> int:
    """
    Count meaningful reasoning steps in the think block.
    Steps are indicated by: line breaks with computations, "let me try",
    "so", "therefore", "=", explicit arithmetic.
    """
    if not think_content:
        return 0
    
    step_indicators = [
        r'\d+\s*[+\-*/]\s*\d+\s*=',   # explicit computation: "3 + 5 = 8"
        r'(?:let me|let\'s|I\'ll)\s+try',  # "let me try"
        r'(?:so|therefore|thus|hence)',     # logical connectors
        r'(?:wait|no|actually|hmm)',        # self-correction signals
        r'(?:check|verify|test)',           # verification signals
    ]
    
    count = 0
    for pattern in step_indicators:
        count += len(re.findall(pattern, think_content, re.IGNORECASE))
    
    return count


def has_self_verification(think_content: str) -> bool:
    """Check if the model verifies its own answer in the think block."""
    verification_patterns = [
        r'(?:let me|let\'s)\s+(?:check|verify|confirm)',
        r'(?:checking|verifying)',
        r'does\s+(?:this|that|it)\s+(?:equal|work|give)',
        r'\?\s*(?:yes|no|correct|wrong)',
        r'=\s*\d+\s*[✓✗✔✘]',
    ]
    for pattern in verification_patterns:
        if re.search(pattern, think_content, re.IGNORECASE):
            return True
    return False


# ============================================================
# Core Error Classification
# ============================================================
def classify_error(
    solution_str: str,
    target: int,
    numbers: list,
) -> tuple:
    """
    Classify the type of error in the model's response.
    
    Returns:
        (error_type: str, details: dict)
    """
    details = {
        "answer_expr": None,
        "eval_result": None,
        "numbers_used": [],
        "numbers_expected": sorted(numbers),
        "target": target,
        "proximity": 0.0,       # How close to target (0-1)
        "number_overlap": 0.0,  # Fraction of correct numbers used
    }
    
    # Check for empty/garbage
    if not solution_str or len(solution_str.strip()) < 5:
        return ErrorType.EMPTY, details
    
    # Extract answer
    answer_expr = extract_answer(solution_str)
    if answer_expr is None:
        return ErrorType.NO_ANSWER, details
    
    details["answer_expr"] = answer_expr
    
    # Try to parse the expression
    if not answer_expr.strip():
        return ErrorType.FORMAT_ERROR, details
    
    # Evaluate the expression
    eval_result = safe_eval_expr(answer_expr)
    if eval_result is None:
        return ErrorType.FORMAT_ERROR, details
    
    details["eval_result"] = eval_result
    
    # Extract numbers used in the expression
    numbers_used = extract_numbers_from_expr(answer_expr)
    details["numbers_used"] = sorted(numbers_used)
    
    # Check number usage
    expected_sorted = sorted(numbers)
    used_sorted = sorted(numbers_used)
    
    # Calculate number overlap
    expected_multiset = list(expected_sorted)
    overlap_count = 0
    for n in used_sorted:
        if n in expected_multiset:
            expected_multiset.remove(n)
            overlap_count += 1
    details["number_overlap"] = overlap_count / len(numbers) if numbers else 0
    
    # Calculate proximity to target
    if eval_result is not None and target != 0:
        relative_error = abs(eval_result - target) / max(abs(target), 1)
        details["proximity"] = max(0, 1 - relative_error)
    elif eval_result is not None and target == 0:
        details["proximity"] = max(0, 1 - abs(eval_result))
    
    # Classify
    numbers_correct = (used_sorted == expected_sorted)
    result_correct = (eval_result is not None and 
                      abs(eval_result - target) < 1e-6)
    
    if numbers_correct and result_correct:
        return ErrorType.CORRECT, details
    
    if numbers_correct and not result_correct:
        return ErrorType.ARITHMETIC_ERROR, details
    
    if not numbers_correct and result_correct:
        # Got the right number but used wrong operands - still wrong
        return ErrorType.NUMBER_VIOLATION, details
    
    # Check if it's a partial solution (used subset correctly)
    if overlap_count > 0 and overlap_count < len(numbers):
        # Check if the subset computation is at least valid
        if eval_result is not None:
            return ErrorType.PARTIAL_SOLUTION, details
    
    if overlap_count == 0:
        return ErrorType.NUMBER_VIOLATION, details
    
    return ErrorType.NUMBER_VIOLATION, details


# ============================================================
# NSM Reward Computation
# ============================================================
def compute_nsm_reward(
    error_type: str,
    details: dict,
    think_content: str,
    # Hyperparameters
    proximity_weight: float = 0.3,
    number_overlap_weight: float = 0.2,
    reasoning_bonus_weight: float = 0.1,
    verification_bonus: float = 0.05,
) -> float:
    """
    Compute the Negative Sample Mining reward.
    
    For correct answers: 1.0
    For incorrect answers: a fine-grained score in (0, 1) based on:
      - Error severity (base score)
      - Proximity to correct answer
      - Number overlap ratio
      - Reasoning chain quality bonus
    """
    if error_type == ErrorType.CORRECT:
        return 1.0
    
    # Base score from error type
    base = ERROR_SEVERITY.get(error_type, 0.0)
    
    # Proximity bonus (how close the computed result is to target)
    proximity = details.get("proximity", 0.0) * proximity_weight
    
    # Number overlap bonus (fraction of correct numbers used)
    overlap = details.get("number_overlap", 0.0) * number_overlap_weight
    
    # Reasoning quality bonus
    reasoning_bonus = 0.0
    if think_content:
        n_steps = count_reasoning_steps(think_content)
        # Reward having some reasoning, but cap it
        step_score = min(n_steps / 5.0, 1.0)  # Saturates at 5 steps
        reasoning_bonus = step_score * reasoning_bonus_weight
        
        # Extra bonus for self-verification behavior
        if has_self_verification(think_content):
            reasoning_bonus += verification_bonus
    
    # Combine (ensure we stay in (0, 1) for wrong answers)
    total = base + proximity + overlap + reasoning_bonus
    
    # Clamp to (0, 0.99) for wrong answers — correct answers always get 1.0
    return min(max(total, 0.0), 0.99)


# ============================================================
# Public API — Drop-in replacement for TinyZero
# ============================================================
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict = None,
) -> float:
    """
    NSM reward function for countdown task.
    Drop-in compatible with veRL's reward function interface.
    
    Args:
        data_source: Dataset identifier (e.g., "countdown")
        solution_str: The model's full response string
        ground_truth: Dict with 'target' (int) and 'numbers' (list[int])
        extra_info: Optional dict, can contain:
            - 'reward_mode': 'binary' | 'nsm' | 'nsm_annealed'
            - 'training_step': int (for annealing)
            - 'total_steps': int (for annealing)
            - NSM hyperparameters
    
    Returns:
        float: Reward score
    """
    extra_info = extra_info or {}
    reward_mode = extra_info.get('reward_mode', 'nsm')
    
    target = ground_truth['target']
    numbers = ground_truth['numbers']
    
    # --- Original binary mode (baseline comparison) ---
    if reward_mode == 'binary':
        return _compute_binary_score(solution_str, target, numbers)
    
    # --- NSM mode ---
    # Classify the error
    error_type, details = classify_error(solution_str, target, numbers)
    
    # Analyze reasoning chain
    think_content = extract_think_content(solution_str)
    
    # Compute NSM reward
    nsm_reward = compute_nsm_reward(
        error_type=error_type,
        details=details,
        think_content=think_content,
        proximity_weight=extra_info.get('proximity_weight', 0.3),
        number_overlap_weight=extra_info.get('number_overlap_weight', 0.2),
        reasoning_bonus_weight=extra_info.get('reasoning_bonus_weight', 0.1),
        verification_bonus=extra_info.get('verification_bonus', 0.05),
    )
    
    # --- Annealed mode: gradually shift from NSM to binary ---
    if reward_mode == 'nsm_annealed':
        step = extra_info.get('training_step', 0)
        total = extra_info.get('total_steps', 1000)
        # Linear annealing: start with full NSM, end with mostly binary
        anneal_ratio = min(step / max(total, 1), 1.0)
        binary_reward = 1.0 if error_type == ErrorType.CORRECT else 0.0
        return (1 - anneal_ratio) * nsm_reward + anneal_ratio * binary_reward
    
    return nsm_reward


def _compute_binary_score(
    solution_str: str,
    target: int,
    numbers: list,
) -> float:
    """
    Original TinyZero binary reward (for baseline comparison).
    correct=1.0, format_ok=0.1, else=0.0
    """
    answer_expr = extract_answer(solution_str)
    if answer_expr is None:
        return 0.0
    
    # Format reward
    eval_result = safe_eval_expr(answer_expr)
    if eval_result is None:
        return 0.1  # Has <answer> tags but unparseable
    
    # Check correctness
    numbers_used = sorted(extract_numbers_from_expr(answer_expr))
    numbers_expected = sorted(numbers)
    
    if (numbers_used == numbers_expected and
            abs(eval_result - target) < 1e-6):
        return 1.0
    
    return 0.1


# ============================================================
# Detailed Logging (for analysis & paper figures)
# ============================================================
def compute_score_with_details(
    data_source: str,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict = None,
) -> dict:
    """
    Extended version that returns detailed breakdown for analysis.
    Use this for logging/wandb, not for training.
    """
    extra_info = extra_info or {}
    target = ground_truth['target']
    numbers = ground_truth['numbers']
    
    error_type, details = classify_error(solution_str, target, numbers)
    think_content = extract_think_content(solution_str)
    
    nsm_reward = compute_nsm_reward(error_type, details, think_content)
    binary_reward = _compute_binary_score(solution_str, target, numbers)
    
    return {
        "score": nsm_reward,
        "binary_score": binary_reward,
        "error_type": error_type,
        "proximity": details["proximity"],
        "number_overlap": details["number_overlap"],
        "eval_result": details["eval_result"],
        "answer_expr": details["answer_expr"],
        "n_reasoning_steps": count_reasoning_steps(think_content),
        "has_self_verification": has_self_verification(think_content),
        "think_length": len(think_content),
    }
