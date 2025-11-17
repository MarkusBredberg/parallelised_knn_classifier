#!/usr/bin/env python3
"""
Profile serial KNN to identify bottlenecks
"""
import cProfile
import pstats
import numpy as np
from io import StringIO

def profile_serial_knn():
    """Profile the serial KNN implementation"""
    
    # Load data
    data = np.load('produced_data/train_embeddings.npz')
    embeddings = data['embeddings']
    labels = data['labels']
    
    test_data = np.load('produced_data/test_embeddings.npz')
    test_queries = test_data['embeddings']
    
    # Profile KNN classification
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Classify queries
    k = 5
    for query in test_queries:
        # Distance calculation (this should be the bottleneck)
        distances = np.linalg.norm(embeddings - query, axis=1)
        k_nearest = np.argpartition(distances, k)[:k]
        k_labels = labels[k_nearest]
        prediction = np.bincount(k_labels).argmax()
    
    profiler.disable()
    
    # Print stats
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats()
    print(s.getvalue())
    
    # Calculate parallel fraction
    total_time = ps.total_ttime
    distance_time = 0.0
    for func in ps.stats:
        if 'linalg/norm' in func[2]:
            distance_time += ps.stats[func][2]
    parallel_fraction = distance_time / total_time
    print(f"Estimated parallel fraction: {parallel_fraction:.2f}")

if __name__ == "__main__":
    profile_serial_knn()