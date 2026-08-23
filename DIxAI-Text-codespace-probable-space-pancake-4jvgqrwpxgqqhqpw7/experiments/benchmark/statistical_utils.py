"""
Statistical Rigor Utilities for TNNLS-Level Experiments

This module provides utilities for:
- Multi-seed experiment execution
- Statistical significance testing (Wilcoxon, paired t-test)
- Confidence interval computation
- Mean ± Std formatting
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Callable, Any, Optional
import functools
from dataclasses import dataclass
import warnings


@dataclass
class StatisticalResult:
    """Container for statistical test results."""
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    n_samples: int
    
    def __str__(self) -> str:
        return f"{self.mean:.3f} ± {self.std:.3f}"
    
    def __format__(self, format_spec: str) -> str:
        """Support for f-string formatting of the mean value."""
        return format(self.mean, format_spec)
    
    def to_latex(self, bold: bool = False) -> str:
        """Format as LaTeX string."""
        result = f"{self.mean:.3f} $\\pm$ {self.std:.3f}"
        if bold:
            result = f"\\textbf{{{result}}}"
        return result


def compute_statistics(values: List[float], confidence: float = 0.95) -> StatisticalResult:
    """
    Compute mean, std, and confidence interval for a list of values.
    
    Args:
        values: List of values from multiple runs
        confidence: Confidence level for CI (default: 0.95)
    
    Returns:
        StatisticalResult with mean, std, and CI bounds
    """
    values = np.array(values)
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1) if n > 1 else 0.0
    
    # Compute confidence interval using t-distribution
    if n > 1:
        se = std / np.sqrt(n)
        t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin = t_critical * se
        ci_lower = mean - margin
        ci_upper = mean + margin
    else:
        ci_lower = ci_upper = mean
    
    return StatisticalResult(
        mean=mean,
        std=std,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_samples=n
    )


def wilcoxon_test(a: List[float], b: List[float], 
                  alternative: str = 'two-sided') -> Tuple[float, float]:
    """
    Perform Wilcoxon signed-rank test for paired samples.
    
    Args:
        a: First set of measurements
        b: Second set of measurements (paired with a)
        alternative: 'two-sided', 'greater', or 'less'
    
    Returns:
        Tuple of (statistic, p-value)
    """
    a, b = np.array(a), np.array(b)
    
    if len(a) != len(b):
        raise ValueError("Arrays must have the same length for paired test")
    
    if len(a) < 5:
        warnings.warn("Wilcoxon test may be unreliable with fewer than 5 samples")
    
    try:
        statistic, pvalue = stats.wilcoxon(a, b, alternative=alternative)
    except ValueError:
        # All differences are zero or too few samples
        statistic, pvalue = np.nan, 1.0
    
    return statistic, pvalue


def paired_ttest(a: List[float], b: List[float], 
                 alternative: str = 'two-sided') -> Tuple[float, float]:
    """
    Perform paired t-test for comparing two methods.
    
    Args:
        a: First set of measurements
        b: Second set of measurements (paired with a)
        alternative: 'two-sided', 'greater', or 'less'
    
    Returns:
        Tuple of (t-statistic, p-value)
    """
    a, b = np.array(a), np.array(b)
    
    if len(a) != len(b):
        raise ValueError("Arrays must have the same length for paired test")
    
    statistic, pvalue = stats.ttest_rel(a, b, alternative=alternative)
    
    return statistic, pvalue


def format_pvalue(pvalue: float) -> str:
    """Format p-value for display."""
    if pvalue < 0.001:
        return "p < 0.001"
    elif pvalue < 0.01:
        return f"p < 0.01"
    elif pvalue < 0.05:
        return f"p < 0.05"
    else:
        return f"p = {pvalue:.3f}"


def is_significant(pvalue: float, alpha: float = 0.05) -> bool:
    """Check if result is statistically significant."""
    return pvalue < alpha


class MultiSeedRunner:
    """
    Run experiments with multiple random seeds for statistical rigor.
    
    Example usage:
        runner = MultiSeedRunner(seeds=[1, 2, 3, 4, 5])
        results = runner.run(my_experiment_fn, arg1=value1, arg2=value2)
    """
    
    def __init__(self, seeds: List[int] = None, n_seeds: int = 5):
        """
        Initialize runner with specific seeds or generate them.
        
        Args:
            seeds: List of specific seeds to use
            n_seeds: Number of seeds to generate if seeds not provided
        """
        if seeds is not None:
            self.seeds = seeds
        else:
            self.seeds = list(range(42, 42 + n_seeds))
    
    def run(self, experiment_fn: Callable[..., Dict[str, float]], 
            *args, **kwargs) -> Dict[str, StatisticalResult]:
        """
        Run experiment across all seeds and aggregate results.
        
        Args:
            experiment_fn: Function that takes a 'seed' kwarg and returns
                          a dict of metric_name -> value
            *args, **kwargs: Arguments to pass to experiment_fn
        
        Returns:
            Dict mapping metric names to StatisticalResult objects
        """
        all_results: Dict[str, List[float]] = {}
        
        for seed in self.seeds:
            result = experiment_fn(*args, seed=seed, **kwargs)
            
            for metric, value in result.items():
                if metric not in all_results:
                    all_results[metric] = []
                all_results[metric].append(value)
        
        # Compute statistics for each metric
        stats_results = {}
        for metric, values in all_results.items():
            stats_results[metric] = compute_statistics(values)
        
        return stats_results
    
    def run_comparison(self, 
                       method_a_fn: Callable, 
                       method_b_fn: Callable,
                       *args, **kwargs) -> Dict[str, Dict]:
        """
        Run two methods and compare them statistically.
        
        Returns:
            Dict with 'method_a', 'method_b', 'wilcoxon', 'ttest' results
        """
        results_a: Dict[str, List[float]] = {}
        results_b: Dict[str, List[float]] = {}
        
        for seed in self.seeds:
            result_a = method_a_fn(*args, seed=seed, **kwargs)
            result_b = method_b_fn(*args, seed=seed, **kwargs)
            
            for metric, value in result_a.items():
                if metric not in results_a:
                    results_a[metric] = []
                results_a[metric].append(value)
            
            for metric, value in result_b.items():
                if metric not in results_b:
                    results_b[metric] = []
                results_b[metric].append(value)
        
        comparison = {}
        for metric in results_a.keys():
            if metric in results_b:
                a_vals = results_a[metric]
                b_vals = results_b[metric]
                
                _, wilcoxon_p = wilcoxon_test(a_vals, b_vals)
                _, ttest_p = paired_ttest(a_vals, b_vals)
                
                comparison[metric] = {
                    'method_a': compute_statistics(a_vals),
                    'method_b': compute_statistics(b_vals),
                    'wilcoxon_p': wilcoxon_p,
                    'ttest_p': ttest_p,
                    'a_better': np.mean(a_vals) > np.mean(b_vals),
                    'significant': is_significant(wilcoxon_p)
                }
        
        return comparison


def generate_latex_comparison_table(
    methods: Dict[str, Dict[str, StatisticalResult]],
    metrics: List[str],
    method_names: Optional[List[str]] = None,
    caption: str = "Comparison of methods",
    label: str = "tab:comparison"
) -> str:
    """
    Generate a LaTeX table comparing multiple methods.
    
    Args:
        methods: Dict mapping method_key -> {metric_name -> StatisticalResult}
        metrics: List of metric names to include
        method_names: Optional display names for methods
        caption: Table caption
        label: Table label
    
    Returns:
        LaTeX table string
    """
    if method_names is None:
        method_names = list(methods.keys())
    
    # Find best method for each metric
    best_per_metric = {}
    for metric in metrics:
        best_val = -np.inf
        best_method = None
        for method_key, results in methods.items():
            if metric in results:
                if results[metric].mean > best_val:
                    best_val = results[metric].mean
                    best_method = method_key
        best_per_metric[metric] = best_method
    
    # Build table
    n_metrics = len(metrics)
    header = " & ".join(["\\textbf{Method}"] + [f"\\textbf{{{m}}}" for m in metrics])
    
    rows = []
    for method_key, display_name in zip(methods.keys(), method_names):
        cols = [display_name]
        for metric in metrics:
            if metric in methods[method_key]:
                result = methods[method_key][metric]
                is_best = (best_per_metric[metric] == method_key)
                cols.append(result.to_latex(bold=is_best))
            else:
                cols.append("--")
        rows.append(" & ".join(cols) + " \\\\")
    
    table = f"""\\begin{{table}}[t]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{@{{}}l{"c" * n_metrics}@{{}}}}
\\toprule
{header} \\\\ \\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    
    return table


# Export
__all__ = [
    'StatisticalResult',
    'compute_statistics',
    'wilcoxon_test',
    'paired_ttest',
    'format_pvalue',
    'is_significant',
    'MultiSeedRunner',
    'generate_latex_comparison_table',
]
