#!/usr/bin/env python3
"""
Profile serial KNN to identify bottlenecks
Run this AFTER data generation but BEFORE parallel experiments
"""
import cProfile
import pstats
import numpy as np
from io import StringIO
import os
import sys

# ✅ Use same path handling as other scripts
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def profile_serial_knn(out_dir=None):
    """Profile the serial KNN implementation"""
    
    # ✅ Use flexible path
    if out_dir is None:
        out_dir = os.path.join(PROJECT_ROOT, 'produced_data')
    
    # ✅ Check if data exists before profiling
    train_path = f'{out_dir}/strong_scaling/train_embeddings.npz'
    test_path = f'{out_dir}/strong_scaling/test_embeddings.npz'
    
    if not os.path.exists(train_path):
        print(f"✗ Training data not found: {train_path}")
        print("Run data generation first:")
        print("  python step1_generate_features.py")
        sys.exit(1)
    
    if not os.path.exists(test_path):
        print(f"✗ Test data not found: {test_path}")
        print("Run data generation first:")
        print("  python step1_generate_features.py")
        sys.exit(1)
    
    print("="*60)
    print("PROFILING SERIAL KNN")
    print("="*60)
    
    # Load data
    print(f"\nLoading data from: {out_dir}")
    data = np.load(train_path)
    embeddings = data['embeddings']
    labels = data['labels']
    
    test_data = np.load(test_path)
    test_queries = test_data['embeddings']
    
    print(f"Training points: {len(embeddings)}")
    print(f"Test queries: {len(test_queries)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # ✅ Limit test queries for reasonable profiling time
    # Full dataset might take too long
    n_queries = min(100, len(test_queries))
    test_queries = test_queries[:n_queries]
    print(f"\nProfiling with {n_queries} queries (limited for speed)")
    
    # Profile KNN classification
    print("\nStarting profiling...")
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Classify queries
    k = 5
    predictions = []
    for query in test_queries:
        # Distance calculation (this should be the bottleneck)
        distances = np.linalg.norm(embeddings - query, axis=1)
        k_nearest = np.argpartition(distances, k)[:k]
        k_labels = labels[k_nearest]
        prediction = np.bincount(k_labels).argmax()
        predictions.append(prediction)
    
    profiler.disable()
    
    print(f"Profiling complete!\n")
    
    # Print stats
    print("="*60)
    print("PROFILING RESULTS")
    print("="*60)
    
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # ✅ Show top 20 functions
    print(s.getvalue())
    
    # Calculate parallel fraction (Amdahl's Law analysis)
    print("\n" + "="*60)
    print("PARALLELIZATION POTENTIAL (Amdahl's Law)")
    print("="*60)
    
    total_time = ps.total_tt
    
    # ✅ More robust function name matching
    distance_time = 0.0
    for func_key, func_stats in ps.stats.items():
        func_name = func_key[2]  # Function name is third element
        # Look for numpy distance calculations
        if any(keyword in func_name.lower() for keyword in 
               ['norm', 'subtract', 'square', 'linalg']):
            distance_time += func_stats[3]  # Cumulative time
    
    parallel_fraction = distance_time / total_time if total_time > 0 else 0
    
    print(f"\nTotal execution time: {total_time:.3f} seconds")
    print(f"Time in distance calculations: {distance_time:.3f} seconds")
    print(f"Estimated parallel fraction (P): {parallel_fraction:.2%}")
    print(f"Serial fraction (1-P): {(1-parallel_fraction):.2%}")
    
    # ✅ Amdahl's Law predictions
    print("\nAmdahl's Law Speedup Predictions:")
    print("-" * 60)
    for n_procs in [2, 4, 8, 16]:
        theoretical_speedup = 1 / ((1 - parallel_fraction) + parallel_fraction / n_procs)
        print(f"  {n_procs:2d} processes: {theoretical_speedup:.2f}x speedup")
    
    print("\n" + "="*60)
    print("INTERPRETATION:")
    print("="*60)
    print(f"• Distance calculations are {parallel_fraction:.0%} of total time")
    print(f"• This is the parallelizable portion")
    print(f"• Serial overhead ({(1-parallel_fraction):.0%}) limits max speedup")
    print(f"• Max theoretical speedup: {1/(1-parallel_fraction):.1f}x")
    print("\nNote: Actual parallel speedup may differ due to:")
    print("  - Communication overhead (MPI sendrecv)")
    print("  - Load imbalance")
    print("  - Ghost region overhead")
    print("=" * 60)

if __name__ == "__main__":
    # ✅ Support command-line out-dir argument
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Profile serial KNN to identify bottlenecks'
    )
    parser.add_argument('--out-dir', type=str, 
                       default=os.path.join(PROJECT_ROOT, 'produced_data'),
                       help='Output directory (default: ./produced_data)')
    
    args = parser.parse_args()
    
    profile_serial_knn(out_dir=args.out_dir)