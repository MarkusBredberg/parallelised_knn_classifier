#!/usr/bin/env python3
"""
Performance Analysis and Plotting Script
Reads results from parallel_knn_results.json and creates plots
Supports both strong scaling and weak scaling analysis
Visualization function that creates dual-colored plots showing:
- Training data with domain decomposition and ghost regions
- Test data as larger squares with:
  * Inner color = true class
  * Border color = predicted class
"""

import json
import numpy as np
from collections import defaultdict
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

def load_results(out_dir='./produced_data'):
    """
    Load results from JSON files
    """
    print("="*60)
    print("LOADING RESULTS")
    print("="*60)
    
    try:
        with open(f'{out_dir}/parallel_knn_results.json', 'r') as f:
            results = json.load(f)
        print(f"✓ Loaded {len(results)} result entries from {out_dir}")
    except FileNotFoundError:
        print(f"✗ {out_dir}/parallel_knn_results.json not found!")
        return None, None
    
    try:
        with open(f'{out_dir}/parallel_knn_statistics.json', 'r') as f:
            statistics = json.load(f)
        print(f"✓ Loaded {len(statistics)} statistics entries")
    except FileNotFoundError:
        print(f"⚠ {out_dir}/parallel_knn_statistics.json not found!")
        statistics = None
    
    return results, statistics


def aggregate_by_processes(results):
    """
    Group results by number of processes and take most recent
    """
    by_procs = defaultdict(list)
    
    for result in results:
        by_procs[result['n_processes']].append(result)
    
    # Take most recent result for each process count
    aggregated = {}
    for n_procs, entries in by_procs.items():
        # Sort by timestamp and take latest
        latest = sorted(entries, key=lambda x: x['timestamp'])[-1]
        aggregated[n_procs] = latest
    
    return aggregated

def plot_speedup(aggregated_results, figure_dir='./figures', prefix=''):
    """
    Create speedup plot with algorithmic speedup consideration
    """
    print("\nCreating speedup plot...")
    
    # Sort by process count
    procs = sorted(aggregated_results.keys())
    times = [aggregated_results[p]['classification_time'] for p in procs]
    
    # Calculate speedup relative to single process
    if 1 not in procs:
        print("⚠ Warning: No single-process baseline found. Using smallest process count.")
        baseline_time = times[0]
    else:
        baseline_time = aggregated_results[1]['classification_time']
    
    speedups = [baseline_time / t for t in times]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot measured speedup
    ax.plot(procs, speedups, 'o-', linewidth=3, markersize=12, 
            label='Measured Speedup', color='#2E86AB', zorder=3)
    
    # Plot ideal linear speedup (naive parallel scaling only)
    ideal_speedup = list(procs)  # Ideal speedup = P

    ax.plot(ideal_speedup, ideal_speedup, '--', linewidth=2.5, 
            label='Naive parallelisation speedup (linear)', color='gray', alpha=0.6, zorder=1)    
    
    # Calculate algorithmical speedup accounting for reduced search space
    # This already combines both parallel execution AND algorithmic speedup!
    algorithmical_speedup = []
    for p in procs:
        local_fraction = 1.0 / p
        ghost_fraction = 0.15 * 2 * (1.0 / p)  # Ghost regions from both neighbors
        total_search_fraction = min(1.0, local_fraction + ghost_fraction)
        algorithmical_speedup.append(1.0 / total_search_fraction)
    
    ax.plot(procs, algorithmical_speedup, 's--', linewidth=2.5, markersize=8,
            label='Maximal algorithmical speedup O(N/P)', color='#F18F01', alpha=0.8, zorder=2)

    combined_speedup = []
    for i in range(len(procs)):
        combined_p = ideal_speedup[i] * algorithmical_speedup[i]
        combined_speedup.append(combined_p)
    ax.plot(procs, combined_speedup, 'd--', linewidth=2.5, markersize=8,
            label='Combined speedup (parallel + algorithmical)', color='#9B1B30', alpha=0.8, zorder=2)
    
    # Add data labels for measured speedup
    for p, s in zip(procs, speedups):
        ax.annotate(f'{s:.2f}x', 
                   xy=(p, s), 
                   xytext=(5, 8), 
                   textcoords='offset points',
                   fontsize=11,
                   fontweight='bold',
                   color='#2E86AB')
    
    ax.set_xlabel('Number of Processes', fontsize=14, fontweight='bold')
    ax.set_ylabel('Speedup', fontsize=14, fontweight='bold')
    ax.set_title('Parallel KNN Speedup Analysis', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(procs)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    filename = f'{figure_dir}/{prefix}speedup_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()
    
    return procs, speedups, times


def plot_efficiency(procs, speedups, figure_dir='./figures', prefix=''):
    """
    Create efficiency plot with algorithmical efficiency line
    """
    print("Creating efficiency plot...")
    
    # Calculate measured efficiency
    efficiencies = [100 * s / p for p, s in zip(procs, speedups)]
    
    # Calculate algorithmical efficiency accounting for O(N/P) search space
    # algorithmical speedup = P / 1.3, so algorithmical efficiency = 100 / 1.3 ≈ 76.9%
    algorithmical_efficiencies = [100.0 / 1.3 for _ in procs]  # Constant ~76.9%
    
    # Combined efficiency: combined_speedup / P * 100 = (P * (P/1.3)) / P * 100 = P/1.3 * 100
    combined_efficiencies = [p / 1.3 * 100.0 for p in procs]  # Grows linearly with P

    # The algorithmical and combined efficiencies must of course start at 100% at P=1
    algorithmical_efficiencies[0] = 100.0
    combined_efficiencies[0] = 100.0
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot measured efficiency
    ax.plot(procs, efficiencies, 'o-', linewidth=3, markersize=12, color='#2E86AB', zorder=3)
    
    # Plot ideal efficiency (100%)
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=2.5, alpha=0.6, 
               label='Naive parallel efficiency (100%)', zorder=1)
    
    # Plot algorithmical efficiency line
    ax.plot(procs, algorithmical_efficiencies, 's--', linewidth=2.5, markersize=8,
            label='Maximal algorithmical efficiency O(N/P) (~76.9%)', color='#F18F01', alpha=0.8, zorder=2)
    
    # Combined efficiency line grows linearly with P
    ax.plot(procs, combined_efficiencies, 'd--', linewidth=2.5, markersize=8,
            label='Combined efficiency (parallel + algorithmical)', color='#9B1B30', alpha=0.8, zorder=2)
    
    # Add data labels for measured efficiency
    for p, e in zip(procs, efficiencies):
        ax.annotate(f'{e:.1f}%', 
                   xy=(p, e), 
                   xytext=(5, 8), 
                   textcoords='offset points',
                   fontsize=11,
                   fontweight='bold',
                   color='#2E86AB')
    
    ax.set_xlabel('Number of Processes', fontsize=14, fontweight='bold')
    ax.set_ylabel('Efficiency (%)', fontsize=14, fontweight='bold')
    ax.set_title('Parallel KNN Efficiency Analysis\n(With Algorithmic Speedup)', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(procs)
    
    # Adjust y-axis to show all lines clearly
    max_y = max(max(efficiencies), max(combined_efficiencies))
    ax.set_ylim(bottom=0, top=max(max_y * 1.15, 120))

    plt.tight_layout()
    filename = f'{figure_dir}/{prefix}efficiency_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def plot_execution_time(procs, times, figure_dir='./figures', prefix=''):
    """
    Create execution time plot with theoretical expectations
    """
    print("Creating execution time plot...")
    
    # Get baseline time (time at P=1 or smallest P)
    if 1 in procs:
        baseline_idx = procs.index(1)
        baseline_time = times[baseline_idx]
    else:
        baseline_time = times[0]
    
    # Calculate theoretical time curves
    naive_parallel_time = [baseline_time / p for p in procs]  # Ideal: T_1 / P
    algorithmic_time = [baseline_time * 1.3 / p for p in procs]  # O(N/P) search: T_1 * 1.3 / P
    combined_time = [baseline_time * 1.3 / (p * p) for p in procs]  # Combined: T_1 * 1.3 / P²
    
    # Actually, the algorithmic and compbined must start at the baseline time at P=1
    algorithmic_time[0] = baseline_time
    combined_time[0] = baseline_time
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot measured execution time (blue, consistent with speedup plot)
    ax.plot(procs, times, 'o-', linewidth=3, markersize=12, 
            label='Measured Time', color='#2E86AB', zorder=3)
    
    # Plot naive parallel expectation (gray dashed)
    ax.plot(procs, naive_parallel_time, '--', linewidth=2.5, 
            label='Naive parallel scaling (T₁/P)', color='gray', alpha=0.6, zorder=1)
    
    # Plot algorithmic expectation (orange dashed)
    ax.plot(procs, algorithmic_time, 's--', linewidth=2.5, markersize=8,
            label='Algorithmic scaling O(N/P) (T₁·1.3/P)', color='#F18F01', alpha=0.8, zorder=2)
    
    # Plot combined expectation (red dashed)
    ax.plot(procs, combined_time, 'd--', linewidth=2.5, markersize=8,
            label='Combined (T₁·1.3/P²)', color='#9B1B30', alpha=0.8, zorder=2)
    
    # Add data labels for measured time
    for p, t in zip(procs, times):
        ax.annotate(f'{t:.3f}s', 
                   xy=(p, t), 
                   xytext=(5, 8), 
                   textcoords='offset points',
                   fontsize=11,
                   fontweight='bold',
                   color='#2E86AB')
    
    ax.set_xlabel('Number of Processes', fontsize=14, fontweight='bold')
    ax.set_ylabel('Execution Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_title('Execution Time vs Number of Processes', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(procs)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    filename = f'{figure_dir}/{prefix}execution_time.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def plot_accuracy(aggregated_results, figure_dir='./figures', prefix=''):
    """
    Create accuracy plot across different process counts
    """
    print("Creating accuracy plot...")
    
    procs = sorted(aggregated_results.keys())
    accuracies = [aggregated_results[p]['accuracy'] * 100 for p in procs]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(procs, accuracies, 'o-', linewidth=2, markersize=10, color='#6A4C93')
    ax.axhline(y=100, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Naive optimum (100%)')
    
    # Add data labels
    for p, acc in zip(procs, accuracies):
        ax.annotate(f'{acc:.2f}%', 
                   xy=(p, acc), 
                   xytext=(5, 5), 
                   textcoords='offset points',
                   fontsize=10,
                   fontweight='bold')
    
    ax.set_xlabel('Number of Processes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Classification Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Classification Accuracy vs Number of Processes', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(procs)
    
    # Set y-axis range to make variations visible
    if accuracies:
        min_acc = min(accuracies)
        max_acc = max(accuracies)
        margin = 5  # 5% margin
        ax.set_ylim(bottom=max(0, min_acc - margin), top=min(105, max_acc + margin))
    
    plt.tight_layout()
    filename = f'{figure_dir}/{prefix}accuracy_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def plot_load_balance(statistics, figure_dir='./figures', prefix=''):
    """
    Create load balance analysis plot
    """
    if statistics is None:
        print("⚠ Skipping load balance plot (no statistics)")
        return
    
    print("Creating load balance plot...")
    
    # Aggregate by process count
    by_procs = defaultdict(list)
    for stat in statistics:
        by_procs[stat['n_processes']].append(stat)
    
    # Take most recent
    aggregated = {}
    for n_procs, entries in by_procs.items():
        latest = sorted(entries, key=lambda x: x['timestamp'])[-1]
        aggregated[n_procs] = latest
    
    procs = sorted(aggregated.keys())
    imbalance_ratios = [aggregated[p]['imbalance_ratio'] for p in procs]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Imbalance ratio
    ax1.plot(procs, imbalance_ratios, 'o-', linewidth=2, markersize=10, color='#D62828', label='Imbalance Ratio')
    ax1.axhline(y=1.0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Perfect Balance')
    
    for p, r in zip(procs, imbalance_ratios):
        # Only annotate if value is valid
        if np.isfinite(r):
            ax1.annotate(f'{r:.2f}', 
                        xy=(p, r), 
                        xytext=(5, 5), 
                        textcoords='offset points',
                        fontsize=10,
                        fontweight='bold')
    
    ax1.set_xlabel('Number of Processes', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Load Imbalance Ratio', fontsize=12, fontweight='bold')
    ax1.set_title('Load Imbalance vs Number of Processes', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(procs)
    
    # Set y-limits with proper handling of invalid values
    valid_ratios = [r for r in imbalance_ratios if np.isfinite(r)]
    if valid_ratios:
        max_ratio = max(valid_ratios)
        ax1.set_ylim(bottom=0, top=max_ratio * 1.2)
    else:
        ax1.set_ylim(bottom=0, top=2.0)  # Default reasonable range
    
    # Plot 2: Communication overhead
    boundary_pcts = []
    for p in procs:
        stat = aggregated[p]
        total = stat['local_queries'] + stat['boundary_queries']
        pct = 100 * stat['boundary_queries'] / total if total > 0 else 0
        boundary_pcts.append(pct)
    
    ax2.plot(procs, boundary_pcts, 'o-', linewidth=2, markersize=10, color='#F77F00')
    
    for p, pct in zip(procs, boundary_pcts):
        ax2.annotate(f'{pct:.1f}%', 
                    xy=(p, pct), 
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold')
    
    ax2.set_xlabel('Number of Processes', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Boundary Queries (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Communication Overhead\n(% of queries requiring MPI)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xticks(procs)
    ax2.set_ylim(bottom=0, top=100)
    
    plt.tight_layout()
    filename = f'{figure_dir}/{prefix}load_balance_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def plot_weak_scaling(aggregated_results, figure_dir='./figures'):
    """
    Create weak scaling efficiency plot with algorithmic speedup consideration
    """
    print("\nCreating weak scaling plot...")
    
    procs = sorted(aggregated_results.keys())
    times = [aggregated_results[p]['classification_time'] for p in procs]
    queries_per_proc = [aggregated_results[p]['queries_per_process'] for p in procs]
    
    # Calculate weak scaling efficiency
    if 1 in procs:
        baseline_time = aggregated_results[1]['classification_time']
    else:
        baseline_time = times[0]
    
    efficiencies = [100 * baseline_time / t for t in times]
    
    # Calculate EXPECTED efficiency based on reduced search space
    # Assumption: Training data is uniformly distributed
    # 1 process: Each query searches N training points
    # P processes: Each query searches approximately N/P + 2*ghost_fraction*N training points
    # For simplicity, estimate as N/P with ~15% overlap
    training_points = aggregated_results[1].get('training_points', 1000)
    
    expected_work_fraction = []
    for p in procs:
        # Each process has ~training_points/p local points
        # Plus ~15% ghost overlap from neighbors (both sides for interior processes)
        local_fraction = 1.0 / p
        ghost_fraction = 0.15 * 2 * (1.0 / p)  # Ghost regions from both neighbors
        total_search_fraction = min(1.0, local_fraction + ghost_fraction)
        expected_work_fraction.append(total_search_fraction)
    
    # Expected efficiency accounts for algorithmic speedup from reduced search space
    # Time_p ≈ Time_1 * (work_fraction_p)
    # Efficiency = Time_1 / Time_p = 1 / work_fraction_p
    expected_efficiency = [100.0 / frac for frac in expected_work_fraction]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Plot 1: Execution time (decreasing due to reduced search space)
    ax1.plot(procs, times, 'o-', linewidth=2, markersize=10, color='#2E86AB', 
             label='Measured Time')
    ax1.axhline(y=baseline_time, color='gray', linestyle='--', linewidth=2, 
               alpha=0.5, label=f'Baseline ({baseline_time:.3f}s)')
    
    for p, t in zip(procs, times):
        ax1.annotate(f'{t:.3f}s', 
                    xy=(p, t), 
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold')
    
    ax1.set_xlabel('Number of Processes', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Weak Scaling: Execution Time', 
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(procs)
    ax1.set_ylim(bottom=0)
    
    # Plot 2: Weak scaling efficiency (with algorithmic context)
    ax2.plot(procs, efficiencies, 'o-', linewidth=2, markersize=10, color='#2E86AB',
             label='Measured Efficiency')
    ax2.axhline(y=100, color='green', linestyle='--', linewidth=2, 
               alpha=0.5, label='Perfect (100%)')
    
    for p, e in zip(procs, efficiencies):
        ax2.annotate(f'{e:.1f}%', 
                    xy=(p, e), 
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold')
    
    ax2.set_xlabel('Number of Processes', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Weak Scaling Efficiency (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Weak Scaling Efficiency', 
                 fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='lower right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xticks(procs)
    
    # Fix y-axis to show all points
    max_eff = max(max(efficiencies), max(expected_efficiency))
    ax2.set_ylim(bottom=0, top=min(max_eff * 1.15, 500))  # Cap at 500% to keep reasonable scale
    
    plt.tight_layout()
    filename = f'{figure_dir}/Weak_scaling_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def create_summary_table(aggregated_results, statistics, out_dir='./produced_data', prefix=''):
    """
    Create a summary table in text format
    """
    print("\nCreating summary table...")
    
    procs = sorted(aggregated_results.keys())
    
    # Calculate all metrics
    if 1 in procs:
        baseline_time = aggregated_results[1]['classification_time']
    else:
        baseline_time = aggregated_results[procs[0]]['classification_time']
    
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY TABLE")
    print("="*80)
    print(f"{'Procs':<8} {'Time(s)':<10} {'Speedup':<10} {'Efficiency':<12} {'Imbalance':<12} {'Boundary%':<12}")
    print("-"*80)
    
    rows = []
    for p in procs:
        time_val = aggregated_results[p]['classification_time']
        speedup = baseline_time / time_val
        efficiency = 100 * speedup / p
        
        if statistics:
            stat = [s for s in statistics if s['n_processes'] == p]
            if stat:
                stat = sorted(stat, key=lambda x: x['timestamp'])[-1]
                imbalance = stat['imbalance_ratio']
                total_q = stat['local_queries'] + stat['boundary_queries']
                boundary_pct = 100 * stat['boundary_queries'] / total_q if total_q > 0 else 0
            else:
                imbalance = 0
                boundary_pct = 0
        else:
            imbalance = 0
            boundary_pct = 0
        
        print(f"{p:<8} {time_val:<10.4f} {speedup:<10.2f} {efficiency:<12.1f} {imbalance:<12.2f} {boundary_pct:<12.1f}")
        
        rows.append({
            'processes': p,
            'time': time_val,
            'speedup': speedup,
            'efficiency': efficiency,
            'imbalance': imbalance,
            'boundary_pct': boundary_pct
        })
    
    print("="*80)
    
    # Save to CSV
    csv_filename = f'{out_dir}/{prefix}performance_summary.csv'
    with open(csv_filename, 'w') as f:
        f.write('Processes,Time(s),Speedup,Efficiency(%),Imbalance,Boundary(%)\n')
        for row in rows:
            f.write(f"{row['processes']},{row['time']:.4f},{row['speedup']:.2f},"
                   f"{row['efficiency']:.1f},{row['imbalance']:.2f},{row['boundary_pct']:.1f}\n")
    
    print(f"\n✓ Saved: {csv_filename}")


def analyze_scaling_mode(mode='strong', base_dir='./produced_data'):
    """
    Analyze results for a specific scaling mode
    """
    if mode == 'strong':
        out_dir = f'{base_dir}/strong_scaling'
        prefix = 'strong_'
        print("\n" + "="*70)
        print("ANALYZING STRONG SCALING RESULTS")
        print("="*70)
    else:
        out_dir = f'{base_dir}/weak_scaling'
        prefix = 'weak_'
        print("\n" + "="*70)
        print("ANALYZING WEAK SCALING RESULTS")
        print("="*70)
    
    # Check if directory exists
    if not Path(out_dir).exists():
        print(f"✗ Directory not found: {out_dir}")
        print(f"  Run experiments first with {'--weak-scaling' if mode == 'weak' else ''} flag")
        return False
    
    # Load results
    results, statistics = load_results(out_dir=out_dir)
    
    if results is None:
        return False
    
    # Aggregate by process count
    aggregated_results = aggregate_by_processes(results)
    
    print(f"\nFound results for: {sorted(aggregated_results.keys())} processes")
    
    # Create plots based on scaling mode
    if mode == 'weak':
        plot_weak_scaling(aggregated_results)
    procs, speedups, times = plot_speedup(aggregated_results, prefix=prefix)
    plot_efficiency(procs, speedups, prefix=prefix)
    plot_execution_time(procs, times, prefix=prefix)
    plot_accuracy(aggregated_results, prefix=prefix)
    plot_load_balance(statistics, prefix=prefix)
    
    # Create summary table
    create_summary_table(aggregated_results, statistics, out_dir=out_dir, prefix=prefix)
    
    return True

def visualize_predictions_with_domains(n_procs=4, 
                                       base_dir='./produced_data',
                                       scaling_mode='strong',
                                       balanced_partitioning=False,
                                       figure_dir='./figures'):
    """
    Visualize domain decomposition with test predictions overlay
    
    Creates plots similar to fmap_domain_with_ghosts but adds test data visualization:
    - Training data shown as small points colored by process
    - Domain boundaries and ghost regions shown with lines/shading
    - Test data shown as LARGER SQUARES with dual coloring:
      * Fill color = true class
      * Edge color = predicted class
      * Correct predictions: fill and edge colors match
      * Incorrect predictions: fill and edge colors differ
    
    Parameters:
    -----------
    n_procs : int
        Number of processes used in the run
    base_dir : str
        Base directory containing produced_data
    scaling_mode : str
        'strong' or 'weak'
    balanced_partitioning : bool
        Whether balanced partitioning was used
    figure_dir : str
        Directory to save figures
    """
    # Construct paths based on scaling mode
    if scaling_mode == 'weak':
        data_dir = f'{base_dir}/weak_scaling'
        train_path = f'{data_dir}/train_embeddings.npz'
        pred_path = f'{data_dir}/predictions_{n_procs}procs.npz'
        mode_suffix = 'weak'
    else:
        data_dir = f'{base_dir}/strong_scaling'
        train_path = f'{data_dir}/train_embeddings.npz'
        pred_path = f'{data_dir}/predictions_{n_procs}procs.npz'
        mode_suffix = 'strong'
    
    # Load training data
    try:
        train_data = np.load(train_path)
        train_embeddings = train_data['embeddings']
        train_labels = train_data['labels']
    except FileNotFoundError:
        print(f"✗ Training data not found: {train_path}")
        return
    
    # Load predictions
    try:
        pred_data = np.load(pred_path)
        test_embeddings = pred_data['embeddings']
        test_true_labels = pred_data['true_labels']
        test_predictions = pred_data['predictions']
    except FileNotFoundError:
        print(f"✗ Prediction data not found: {pred_path}")
        print(f"  Run: mpirun -np {n_procs} python step2_parallel_knn.py")
        return
    
    print(f"\n{'='*70}")
    print(f"CREATING PREDICTION VISUALIZATION ({n_procs} processes, {scaling_mode} scaling)")
    print(f"{'='*70}")
    print(f"Training points: {len(train_embeddings)}")
    print(f"Test points: {len(test_embeddings)}")
    
    # Calculate accuracy
    accuracy = np.mean(test_predictions == test_true_labels)
    n_correct = np.sum(test_predictions == test_true_labels)
    n_incorrect = np.sum(test_predictions != test_true_labels)
    print(f"Accuracy: {accuracy*100:.2f}% ({n_correct} correct, {n_incorrect} incorrect)")
    
    # Compute domain boundaries
    min_val = train_embeddings[:, 0].min()
    max_val = train_embeddings[:, 0].max()
    global_width = max_val - min_val
    
    if balanced_partitioning:
        # Data-driven partitioning
        sort_indices = np.argsort(train_embeddings[:, 0])
        sorted_embeddings = train_embeddings[sort_indices]
        
        points_per_proc = len(train_embeddings) // n_procs
        boundaries = [min_val]
        
        for r in range(1, n_procs):
            boundary_idx = r * points_per_proc
            boundary_value = sorted_embeddings[boundary_idx, 0]
            boundaries.append(boundary_value)
        
        boundaries.append(max_val)
        boundaries = np.array(boundaries)
        part_mode = "Balanced (Data-Driven)"
    else:
        # Uniform spatial partitioning
        boundaries = np.linspace(min_val, max_val, n_procs + 1)
        part_mode = "Uniform Spatial"
    
    # Calculate domain widths and ghost widths
    domain_widths = np.diff(boundaries)
    ghost_widths = domain_widths * 0.15  # 15% as in the code

    # Calculate y-axis range for positioning labels (MUST BE BEFORE PLOTTING)
    y_min = min(train_embeddings[:, 1].min(), test_embeddings[:, 1].min())
    y_max = max(train_embeddings[:, 1].max(), test_embeddings[:, 1].max())
    y_range = y_max - y_min

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define colors for processes and classes
    proc_colors = plt.cm.tab10(np.linspace(0, 1, n_procs))

    # Get unique classes and define class colors
    unique_classes = np.unique(np.concatenate([train_labels, test_true_labels]))
    n_classes = len(unique_classes)
    class_colors = plt.cm.Set3(np.linspace(0, 1, n_classes))

    # Create color maps
    class_to_color = {cls: class_colors[i] for i, cls in enumerate(unique_classes)}

    # Assign each training point to a process
    process_ids = np.digitize(train_embeddings[:, 0], boundaries[1:-1])
    
    # =========================================================
    # PART 1: Plot training data colored by class
    # =========================================================
    for proc_id in range(n_procs):
        proc_min = boundaries[proc_id]
        proc_max = boundaries[proc_id + 1]
        proc_center = (proc_min + proc_max) / 2
        
        # Count points
        mask = process_ids == proc_id
        n_points = mask.sum()
        
        # Alternate vertical positions
        if proc_id % 2 == 0:
            label_y = y_max + y_range * 0.18
        else:
            label_y = y_max + y_range * 0.32

        mask = process_ids == proc_id
        ax.scatter(train_embeddings[mask, 0], train_embeddings[mask, 1], 
                  c=[class_to_color[train_labels[i]] for i in np.where(mask)[0]], alpha=0.5, s=30, 
                  label=f'Process {proc_id} (train)', zorder=2)

        # Add text box
        ax.text(proc_center, label_y, 
               f'Process {proc_id}\n{n_points} points', 
               ha='center', va='bottom', fontsize=16, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.6', 
                        alpha=0.85, edgecolor='black', linewidth=2))
    
    # =========================================================
    # PART 2: Draw domain boundaries and ghost regions
    # =========================================================
    for i, boundary in enumerate(boundaries):
        ax.axvline(boundary, color='red', linestyle='-', linewidth=4, 
                  alpha=0.8, zorder=3, label='Domain Boundary' if i == 0 else '')
    
    # Ghost regions
    for proc_id in range(n_procs):
        proc_min = boundaries[proc_id]
        proc_max = boundaries[proc_id + 1]
        ghost_width = ghost_widths[proc_id]
        
        # Left ghost region
        if proc_id > 0:
            left_ghost_boundary = proc_min + ghost_width
            ax.axvline(left_ghost_boundary, color='blue', linestyle='--', 
                      linewidth=3, alpha=0.7, zorder=3,
                      label='Ghost Region Boundary' if proc_id == 1 else '')
            ax.axvspan(proc_min, left_ghost_boundary, alpha=0.12, 
                      color='blue', zorder=1)
        
        # Right ghost region
        if proc_id < n_procs - 1:
            right_ghost_boundary = proc_max - ghost_width
            ax.axvline(right_ghost_boundary, color='blue', linestyle='--', 
                      linewidth=3, alpha=0.7, zorder=3)
            ax.axvspan(right_ghost_boundary, proc_max, alpha=0.12, 
                      color='blue', zorder=1)
    
    # =========================================================
    # PART 3: Plot test data as DUAL-COLORED SQUARES
    # =========================================================
    
    for i, (emb, true_label, pred_label) in enumerate(zip(test_embeddings, 
                                                          test_true_labels, 
                                                          test_predictions)):
        x, y = emb[0], emb[1]
        
        # Base square with true class color
        ax.add_patch(
            mpatches.Rectangle((x - 0.03, y - 0.03), 
                               width=0.06, height=0.12,
                              facecolor=class_to_color[true_label],
                              alpha=0.6,
                              zorder=4)
        )   

        # If wrong. outer edge color = predicted class
        if true_label != pred_label:
            edge_color = class_to_color[pred_label]  # Outer color = predicted class
            ax.add_patch(
                mpatches.Rectangle((x - 0.04, y - 0.04), 
                                   width=0.08, height=0.16,
                                  facecolor='none', 
                                  edgecolor=edge_color,
                                  linewidth=2.0,
                                  zorder=5)
            )

    
    # =========================================================
    # PART 4: Create comprehensive legend
    # =========================================================
    
    # Legend elements for domain decomposition
    boundary_elements = [
        mpatches.Patch(color='red', label='Domain Boundaries (MPI process boundaries)'),
        mpatches.Patch(color='blue', alpha=0.3, label='Ghost Regions (overlapping data)'),
        plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=3, 
                   label='Ghost Region Limits')
    ]
    
    # Legend elements for class colors
    class_elements = [
        mpatches.Patch(facecolor=class_to_color[cls], edgecolor='black', 
                      label=f'Class {cls}')
        for cls in unique_classes
    ]
    
    # Legend elements for prediction interpretation
    test_elements = [
        mpatches.Rectangle((0, 0), 1, 1, 
                          facecolor='lightgray', edgecolor='lightgray',
                          linewidth=1.5,
                          label='Test: Correct (fill = edge color)'),
        mpatches.Rectangle((0, 0), 1, 1, 
                          facecolor='lightgray', edgecolor='red',
                          linewidth=4.0,
                          label='Test: Incorrect (fill ≠ edge color) \n Edge = predicted class. Fill = true class')
    ]
    
    # Create two legends
    legend1 = ax.legend(handles=boundary_elements, loc='lower right', 
                       fontsize=13, framealpha=0.95, edgecolor='black', 
                       title='Domain Decomposition', title_fontsize=14,
                       fancybox=True, shadow=True)
    ax.add_artist(legend1)
    
    legend2 = ax.legend(handles=class_elements + test_elements, 
                       loc='lower left', fontsize=12, ncol=2,
                       framealpha=0.95, edgecolor='black',
                       title='Class Colors & Test Predictions', title_fontsize=13,
                       fancybox=True, shadow=True)
    ax.add_artist(legend2)
    
    
    # =========================================================
    # PART 5: Labels and title
    # =========================================================
    ax.set_xlabel('First Embedding Dimension (Decomposition Axis)', 
                 fontsize=18, fontweight='bold')
    ax.set_ylabel('Second Embedding Dimension', fontsize=18, fontweight='bold')
        
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1.5)
    ax.tick_params(axis='both', labelsize=14)
    
    # Adjust limits to ensure all data visible
    ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.05)
    
    plt.tight_layout()
    
    # Save with descriptive filename
    part_suffix = "balanced" if balanced_partitioning else "uniform"
    filename = f'{figure_dir}/{mode_suffix}_predictions_with_domains_{n_procs}procs_{part_suffix}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()
    
    # Print interpretation guide
    print(f"\n{'='*70}")
    print("INTERPRETATION GUIDE:")
    print(f"{'='*70}")
    print("Training data:")
    print("  • Small circles colored by which process owns them")
    print("  • Shows domain decomposition and ghost regions")
    print("\nTest data (larger squares):")
    print("  • Fill color = TRUE class label")
    print("  • Edge color = PREDICTED class label")
    print("  • Matching colors = correct prediction ✓")
    print("  • Different colors = incorrect prediction ✗")
    print(f"{'='*70}\n")


def create_all_prediction_visualizations(base_dir='./produced_data', 
                                        figure_dir='./figures'):
    """
    Create prediction visualizations for all available process counts
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing produced_data
    figure_dir : str
        Directory to save figures
    """
    from pathlib import Path
    
    print("\n" + "="*70)
    print("CREATING ALL PREDICTION VISUALIZATIONS")
    print("="*70)
    
    # Check for strong scaling
    strong_dir = Path(f'{base_dir}/strong_scaling')
    if strong_dir.exists():
        print("\nStrong Scaling:")
        print("-" * 70)
        
        # Find all prediction files
        pred_files = list(strong_dir.glob('predictions_*procs.npz'))
        
        for pred_file in sorted(pred_files):
            # Extract number of processes from filename
            n_procs = int(pred_file.stem.split('_')[1].replace('procs', ''))
            
            # Try both uniform and balanced
            for balanced in [False, True]:
                try:
                    visualize_predictions_with_domains(
                        n_procs=n_procs,
                        base_dir=base_dir,
                        scaling_mode='strong',
                        balanced_partitioning=balanced,
                        figure_dir=figure_dir
                    )
                except Exception as e:
                    print(f"⚠ Warning: Could not create visualization for {n_procs} procs "
                          f"({'balanced' if balanced else 'uniform'}): {e}")
    
    # Check for weak scaling
    weak_dir = Path(f'{base_dir}/weak_scaling')
    if weak_dir.exists():
        print("\nWeak Scaling:")
        print("-" * 70)
        
        # Find all prediction files
        pred_files = list(weak_dir.glob('predictions_*procs.npz'))
        
        for pred_file in sorted(pred_files):
            # Extract number of processes from filename
            n_procs = int(pred_file.stem.split('_')[1].replace('procs', ''))
            
            # Try both uniform and balanced
            for balanced in [False, True]:
                try:
                    visualize_predictions_with_domains(
                        n_procs=n_procs,
                        base_dir=base_dir,
                        scaling_mode='weak',
                        balanced_partitioning=balanced,
                        figure_dir=figure_dir
                    )
                except Exception as e:
                    print(f"⚠ Warning: Could not create visualization for {n_procs} procs "
                          f"({'balanced' if balanced else 'uniform'}): {e}")
    
    print("\n" + "="*70)
    print("VISUALIZATION GENERATION COMPLETE!")
    print("="*70)

def main():
    """
    Main plotting function - analyzes both strong and weak scaling
    """
    # Check command line flags
    analyze_strong = '--strong' in sys.argv or '--strong-only' in sys.argv
    analyze_weak = '--weak' in sys.argv or '--weak-only' in sys.argv
    
    # If no flags specified, analyze both
    if not analyze_strong and not analyze_weak:
        analyze_strong = True
        analyze_weak = True
    
    base_dir = './produced_data'
    
    print("\n" + "="*70)
    print("PARALLEL KNN PERFORMANCE ANALYSIS")
    print("="*70)
    
    success_count = 0

    create_all_prediction_visualizations()
    
    # Analyze strong scaling
    if analyze_strong:
        if analyze_scaling_mode('strong', base_dir):
            success_count += 1
    
    # Analyze weak scaling
    if analyze_weak:
        if analyze_scaling_mode('weak', base_dir):
            success_count += 1
    
    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    
    if success_count == 0:
        print("\n✗ No results found. Run experiments first:")
        print("  Strong scaling: mpirun -np 1,2,4,8 python step2.1_plot_results.py")
        print("  Weak scaling: mpirun -np 1,2,4,8 python step2.1_plot_results.py --weak-scaling")
    else:
        print("\nGenerated plots:")
        if analyze_strong and Path('./produced_data/strong_scaling').exists():
            print("\nStrong Scaling:")
            print("  • strong_speedup_analysis.png")
            print("  • strong_efficiency_analysis.png")
            print("  • strong_execution_time.png")
            print("  • strong_accuracy_analysis.png")
            print("  • strong_load_balance_analysis.png")
            print("  • strong_performance_summary.csv")
        
        if analyze_weak and Path('./produced_data/weak_scaling').exists():
            print("\nWeak Scaling:")
            print("  • weak_scaling_analysis.png")
            print("  • weak_execution_time.png")
            print("  • weak_performance_summary.csv")
        
        print("\nAll plots saved to: ./figures/")


if __name__ == "__main__":    
    main()