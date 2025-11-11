#!/usr/bin/env python3
"""
Performance Analysis and Plotting Script
Reads results from parallel_knn_results.json and creates plots
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def load_results():
    """
    Load results from JSON files
    """
    print("="*60)
    print("LOADING RESULTS")
    print("="*60)
    
    try:
        with open('parallel_knn_results.json', 'r') as f:
            results = json.load(f)
        print(f"✓ Loaded {len(results)} result entries")
    except FileNotFoundError:
        print("✗ parallel_knn_results.json not found!")
        print("Run: mpirun -np 1,2,4,8 python step2_parallel_knn.py")
        return None, None
    
    try:
        with open('parallel_knn_statistics.json', 'r') as f:
            statistics = json.load(f)
        print(f"✓ Loaded {len(statistics)} statistics entries")
    except FileNotFoundError:
        print("✗ parallel_knn_statistics.json not found!")
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


def plot_speedup(aggregated_results):
    """
    Create speedup plot
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
    ideal_speedup = list(range(1, max(procs) + 1))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(procs, speedups, 'o-', linewidth=2, markersize=10, 
            label='Actual Speedup', color='#2E86AB')
    ax.plot(ideal_speedup, ideal_speedup, '--', linewidth=2, 
            label='Ideal (Linear)', color='#A23B72', alpha=0.7)
    
    # Add data labels
    for p, s in zip(procs, speedups):
        ax.annotate(f'{s:.2f}x', 
                   xy=(p, s), 
                   xytext=(5, 5), 
                   textcoords='offset points',
                   fontsize=10,
                   fontweight='bold')
    
    ax.set_xlabel('Number of Processes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup', fontsize=12, fontweight='bold')
    ax.set_title('Parallel KNN Speedup Analysis', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(procs)
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig('speedup_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: speedup_analysis.png")
    
    return procs, speedups, times


def plot_efficiency(procs, speedups):
    """
    Create efficiency plot
    """
    print("Creating efficiency plot...")
    
    efficiencies = [100 * s / p for p, s in zip(procs, speedups)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(procs, efficiencies, 'o-', linewidth=2, markersize=10, color='#F18F01')
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Ideal (100%)')
    
    # Add data labels
    for p, e in zip(procs, efficiencies):
        ax.annotate(f'{e:.1f}%', 
                   xy=(p, e), 
                   xytext=(5, 5), 
                   textcoords='offset points',
                   fontsize=10,
                   fontweight='bold')
    
    ax.set_xlabel('Number of Processes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Parallel Efficiency (%)', fontsize=12, fontweight='bold')
    ax.set_title('Parallel Efficiency vs Number of Processes', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(procs)
    ax.set_ylim(bottom=0, top=max(120, max(efficiencies) * 1.1))
    
    plt.tight_layout()
    plt.savefig('efficiency_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: efficiency_analysis.png")


def plot_execution_time(procs, times):
    """
    Create execution time plot
    """
    print("Creating execution time plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(procs, times, 'o-', linewidth=2, markersize=10, color='#06A77D')
    
    # Add data labels
    for p, t in zip(procs, times):
        ax.annotate(f'{t:.3f}s', 
                   xy=(p, t), 
                   xytext=(5, 5), 
                   textcoords='offset points',
                   fontsize=10,
                   fontweight='bold')
    
    ax.set_xlabel('Number of Processes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Execution Time vs Number of Processes', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(procs)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig('execution_time.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: execution_time.png")


def plot_load_balance(statistics):
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
    ax1.plot(procs, imbalance_ratios, 'o-', linewidth=2, markersize=10, color='#D62828')
    ax1.axhline(y=1.0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Perfect Balance')
    ax1.axhline(y=2.0, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Warning Threshold')
    
    for p, r in zip(procs, imbalance_ratios):
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
    ax1.set_ylim(bottom=0)
    
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
    plt.savefig('load_balance_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: load_balance_analysis.png")


def create_summary_table(aggregated_results, statistics):
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
    with open('performance_summary.csv', 'w') as f:
        f.write('Processes,Time(s),Speedup,Efficiency(%),Imbalance,Boundary(%)\n')
        for row in rows:
            f.write(f"{row['processes']},{row['time']:.4f},{row['speedup']:.2f},"
                   f"{row['efficiency']:.1f},{row['imbalance']:.2f},{row['boundary_pct']:.1f}\n")
    
    print("\n✓ Saved: performance_summary.csv")


def main():
    """
    Main plotting function
    """
    # Load results
    results, statistics = load_results()
    
    if results is None:
        return
    
    # Aggregate by process count
    aggregated_results = aggregate_by_processes(results)
    
    print(f"\nFound results for: {sorted(aggregated_results.keys())} processes")
    
    # Create all plots
    procs, speedups, times = plot_speedup(aggregated_results)
    plot_efficiency(procs, speedups)
    plot_execution_time(procs, times)
    plot_load_balance(statistics)
    
    # Create summary table
    create_summary_table(aggregated_results, statistics)
    
    print("\n" + "="*60)
    print("PLOTTING COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  1. speedup_analysis.png - Speedup vs processes")
    print("  2. efficiency_analysis.png - Parallel efficiency")
    print("  3. execution_time.png - Execution time")
    print("  4. load_balance_analysis.png - Load imbalance and communication")
    print("  5. performance_summary.csv - All data in table format")
    print("\nUse these in your report!")


if __name__ == "__main__":
    main()