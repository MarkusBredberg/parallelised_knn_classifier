#!/usr/bin/env python3
"""
Step 2: Parallel KNN Classifier with Spatial Decomposition
Operates on embeddings from pre-trained encoder (Step 1)
Uses spatial decomposition of embedding space with MPI communication

UPDATED: Now saves predictions for visualization
"""

import numpy as np
from mpi4py import MPI
import argparse
import json
import datetime
from pathlib import Path    

class ParallelKNN:
    """
    Parallel KNN classifier using spatial decomposition
    Similar structure to your Poisson solver!
    """
    def __init__(self, k=5, n_dims=512, balanced_partitioning=False, comm=MPI.COMM_WORLD):
        """
        Initialize parallel KNN classifier
        
        Parameters:
        -----------
        k : int
            Number of nearest neighbors
        n_dims : int
            Dimensionality of embeddings (512 for BYOL, 2048 for DINO)
        balanced_partitioning : bool
            If True, use data-driven partitioning to balance load
            If False, use uniform spatial partitioning (may cause imbalance with clustered data)
        comm : MPI.Comm
            MPI communicator
        """
        # Core KNN parameters
        self.k = k
        self.n_dims = n_dims
        self.balanced_partitioning = balanced_partitioning
        self.comm = comm
        
        # MPI rank and size
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        
        # Determine process grid layout (1D for simplicity, 2D possible)
        # For now: 1D decomposition along first embedding dimension
        self.n_procs = self.size
        
        # Domain boundaries (will be set when loading data)
        self.my_min_bound = None
        self.my_max_bound = None
        self.domain_width = None
        
        # Neighbor ranks (like m_north_prank, m_south_prank in Poisson)
        self.left_rank = (self.rank - 1) if self.rank > 0 else MPI.PROC_NULL
        self.right_rank = (self.rank + 1) if self.rank < self.size - 1 else MPI.PROC_NULL
        
        # Training data storage
        self.local_training_data = None  # Points in my domain
        self.local_training_labels = None
        self.ghost_training_data = None  # Cached from neighbors
        self.ghost_training_labels = None
        
        # Ghost width (determines communication volume)
        self.ghost_width = None
        
        # Statistics
        self.local_queries = 0
        self.boundary_queries = 0
    
    def load_and_distribute_data(self, embeddings, labels):
        """
        Load embeddings and distribute across processes based on spatial location
        
        Two modes:
        1. Uniform spatial partitioning (balanced_partitioning=False):
        - Divides embedding space uniformly
        - May cause load imbalance with clustered data
        
        2. Data-driven partitioning (balanced_partitioning=True):
        - Uses percentiles to ensure equal number of points per process
        - Better load balance but loses spatial locality
        
        Parameters:
        -----------
        embeddings : np.ndarray (N x n_dims)
            Training embeddings (only on rank 0 initially)
        labels : np.ndarray (N,)
            Training labels (only on rank 0 initially)
        """
        if self.rank == 0:
            print(f"\nLoading embeddings...")
            print(f"Loaded {len(embeddings)} embeddings of dimension {self.n_dims}")
            
            # Verify label distribution on rank 0
            unique_labels, label_counts = np.unique(labels, return_counts=True)
            print(f"\n✓ Label Distribution in Training Data:")
            for label, count in zip(unique_labels, label_counts):
                print(f"  Class {label}: {count} samples ({100*count/len(labels):.1f}%)")
            
            # Compute global domain boundaries based on first dimension
            global_min = embeddings[:, 0].min()
            global_max = embeddings[:, 0].max()
            global_width = global_max - global_min
            
            print(f"\nGlobal embedding space: [{global_min:.4f}, {global_max:.4f}]")
            
            if self.balanced_partitioning:
                print(f"Using DATA-DRIVEN partitioning (equal points per process)")
                
                # Sort data by first dimension
                sort_indices = np.argsort(embeddings[:, 0])
                sorted_embeddings = embeddings[sort_indices]
                sorted_labels = labels[sort_indices]
                
                # Compute percentile boundaries for equal-sized partitions
                points_per_proc = len(embeddings) // self.size
                boundaries = [global_min]  # Start with global min
                
                for r in range(1, self.size):
                    # Find the boundary that splits data evenly
                    boundary_idx = r * points_per_proc
                    boundary_value = sorted_embeddings[boundary_idx, 0]
                    boundaries.append(boundary_value)
                
                boundaries.append(global_max)  # End with global max
                
                print(f"Percentile-based boundaries: {[f'{b:.4f}' for b in boundaries]}")
                
                # Distribute data to all processes
                for r in range(self.size):
                    r_min = boundaries[r]
                    r_max = boundaries[r + 1]
                    r_width = r_max - r_min
                    
                    # Send boundaries
                    bounds = np.array([r_min, r_max, r_width])
                    self.comm.send(bounds, dest=r, tag=10)
                    
                    # Select data in this domain
                    if r < self.size - 1:
                        mask = (sorted_embeddings[:, 0] >= r_min) & (sorted_embeddings[:, 0] < r_max)
                    else:
                        mask = (sorted_embeddings[:, 0] >= r_min) & (sorted_embeddings[:, 0] <= r_max)
                    
                    local_data = sorted_embeddings[mask]
                    local_labels = sorted_labels[mask]
                    
                    if r == 0:
                        self.local_training_data = local_data
                        self.local_training_labels = local_labels
                    else:
                        self.comm.send(local_data, dest=r, tag=0)
                        self.comm.send(local_labels, dest=r, tag=1)
            
            else:
                print(f"Using UNIFORM SPATIAL partitioning (may cause imbalance)")
                
                # Original uniform partitioning
                for r in range(self.size):
                    # Compute domain boundaries for process r
                    r_min = global_min + r * global_width / self.size
                    r_max = global_min + (r + 1) * global_width / self.size
                    
                    # Send boundaries to process r
                    bounds = np.array([r_min, r_max, global_width / self.size])
                    self.comm.send(bounds, dest=r, tag=10)
                    
                    # Select data in this domain
                    if r < self.size - 1:
                        mask = (embeddings[:, 0] >= r_min) & (embeddings[:, 0] < r_max)
                    else:
                        mask = (embeddings[:, 0] >= r_min) & (embeddings[:, 0] <= r_max)
                    
                    local_data = embeddings[mask]
                    local_labels = labels[mask]
                    
                    if r == 0:
                        self.local_training_data = local_data
                        self.local_training_labels = local_labels
                    else:
                        self.comm.send(local_data, dest=r, tag=0)
                        self.comm.send(local_labels, dest=r, tag=1)
        
        else:
            # Non-root processes receive their data
            self.local_training_data = self.comm.recv(source=0, tag=0)
            self.local_training_labels = self.comm.recv(source=0, tag=1)
        
        # All processes receive their domain bounds
        bounds = self.comm.recv(source=0, tag=10)
        self.my_min_bound = bounds[0]
        self.my_max_bound = bounds[1]
        self.domain_width = bounds[2]
        
        # Verify label distribution on each process
        if len(self.local_training_labels) > 0:
            unique_local, local_counts = np.unique(self.local_training_labels, return_counts=True)
            label_str = ", ".join([f"Class {l}: {c}" for l, c in zip(unique_local, local_counts)])
            print(f"Rank {self.rank}: Domain [{self.my_min_bound:.4f}, {self.my_max_bound:.4f}] "
                  f"with {len(self.local_training_data)} training points ({label_str})")
        else:
            print(f"Rank {self.rank}: Domain [{self.my_min_bound:.4f}, {self.my_max_bound:.4f}] "
                  f"with 0 training points ⚠ WARNING: Empty domain!")
        
        # Compute ghost width
        self.ghost_width = self._compute_ghost_width()
    
    def _compute_ghost_width(self):
        """
        Compute ghost region width
        Similar to ghost cell width in Poisson (which is 1 row)
        
        Here: based on k and local data density
        """
        if len(self.local_training_data) == 0:
            return self.domain_width * 0.2  # Default: 20% overlap
        
        # Estimate: distance to k-th nearest neighbor in local domain
        # Conservative: use fixed percentage of domain width
        ghost_width = self.domain_width * 0.15  # 15% overlap with neighbors
        
        print(f"Rank {self.rank}: Ghost width = {ghost_width:.4f}")
        return ghost_width
    
    def exchange_ghost_regions(self):
        """
        Exchange boundary training points with neighbors
        THIS IS LIKE GHOST CELL EXCHANGE IN POISSON SOLVER!
        
        Similar to:
        MPI_Sendrecv(&uo(1, 0), m_local_n, MPI_FLOAT, m_north_prank, ...)
        """
        print(f"Rank {self.rank}: Starting ghost region exchange...")
        
        # Pack boundary points for left neighbor
        left_boundary_mask = (self.local_training_data[:, 0] - self.my_min_bound) < self.ghost_width
        left_boundary_data = self.local_training_data[left_boundary_mask]
        left_boundary_labels = self.local_training_labels[left_boundary_mask]
        
        # Pack boundary points for right neighbor
        right_boundary_mask = (self.my_max_bound - self.local_training_data[:, 0]) < self.ghost_width
        right_boundary_data = self.local_training_data[right_boundary_mask]
        right_boundary_labels = self.local_training_labels[right_boundary_mask]
        
        print(f"Rank {self.rank}: Sending {len(left_boundary_data)} points left, "
              f"{len(right_boundary_data)} points right")
        
        # Exchange with neighbors using sendrecv
        # CRITICAL: When sending right, receive from LEFT (and vice versa)
        # This is exactly like Poisson ghost cell exchange!
        
        # Send right boundary to right neighbor, receive left boundary from left neighbor
        ghost_from_left_data = self.comm.sendrecv(
            sendobj=right_boundary_data,
            dest=self.right_rank,
            source=self.left_rank,  # Receive from LEFT when sending RIGHT
            sendtag=0,
            recvtag=0
        )
        ghost_from_left_labels = self.comm.sendrecv(
            sendobj=right_boundary_labels,
            dest=self.right_rank,
            source=self.left_rank,  # Receive from LEFT when sending RIGHT
            sendtag=1,
            recvtag=1
        )
        
        # Send left boundary to left neighbor, receive right boundary from right neighbor
        ghost_from_right_data = self.comm.sendrecv(
            sendobj=left_boundary_data,
            dest=self.left_rank,
            source=self.right_rank,  # Receive from RIGHT when sending LEFT
            sendtag=2,
            recvtag=2
        )
        ghost_from_right_labels = self.comm.sendrecv(
            sendobj=left_boundary_labels,
            dest=self.left_rank,
            source=self.right_rank,  # Receive from RIGHT when sending LEFT
            sendtag=3,
            recvtag=3
        )
        
        # Combine ghost data from both neighbors
        ghost_data_list = []
        ghost_labels_list = []
        
        if ghost_from_left_data is not None:
            ghost_data_list.append(ghost_from_left_data)
            ghost_labels_list.append(ghost_from_left_labels)
        
        if ghost_from_right_data is not None:
            ghost_data_list.append(ghost_from_right_data)
            ghost_labels_list.append(ghost_from_right_labels)
        
        if ghost_data_list:
            self.ghost_training_data = np.vstack(ghost_data_list)
            self.ghost_training_labels = np.concatenate(ghost_labels_list)
        else:
            self.ghost_training_data = np.empty((0, self.n_dims))
            self.ghost_training_labels = np.empty(0, dtype=int)
    
    def is_boundary_query(self, query):
        """
        Check if a query point is near domain boundary
        Boundary queries require communication with neighbors
        """
        query_x = query[0]
        
        # Check if within ghost_width of boundaries
        near_left = (query_x - self.my_min_bound) < self.ghost_width
        near_right = (self.my_max_bound - query_x) < self.ghost_width
        
        return near_left or near_right
    
    def classify_query(self, query):
        """
        Classify a single query point using KNN
        
        Parameters:
        -----------
        query : np.ndarray (n_dims,)
            Query point to classify
        
        Returns:
        --------
        predicted_label : int
            Predicted class label
        """
        # Combine local and ghost training data
        all_training_data = np.vstack([self.local_training_data, self.ghost_training_data])
        all_training_labels = np.concatenate([self.local_training_labels, self.ghost_training_labels])
        
        # Compute distances to all training points
        distances = np.linalg.norm(all_training_data - query, axis=1)
        
        # Find k nearest neighbors
        k_nearest_indices = np.argpartition(distances, min(self.k, len(distances)-1))[:self.k]
        k_nearest_labels = all_training_labels[k_nearest_indices]
        
        # Majority vote
        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
        predicted_label = unique_labels[np.argmax(counts)]
        
        # Track statistics
        if self.is_boundary_query(query):
            self.boundary_queries += 1
        else:
            self.local_queries += 1
        
        return predicted_label
    
    def classify_batch(self, queries, true_labels=None):
        """
        Classify a batch of query points
        
        Parameters:
        -----------
        queries : np.ndarray (N x n_dims)
            Query points to classify
        true_labels : np.ndarray (N,), optional
            True labels for accuracy computation
        
        Returns:
        --------
        predictions : np.ndarray (N,)
            Predicted labels
        accuracy : float (if true_labels provided)
            Classification accuracy
        """
        predictions = []
        
        start_time = MPI.Wtime()
        
        for query in queries:
            pred = self.classify_query(query)
            predictions.append(pred)
        
        end_time = MPI.Wtime()
        
        predictions = np.array(predictions)
        
        if true_labels is not None:
            accuracy = np.mean(predictions == true_labels)
            return predictions, accuracy, end_time - start_time
        else:
            return predictions, None, end_time - start_time
    
    def print_statistics(self, out_dir='/home/markusbredberg/Scripts/parallelised_knn_classifier/produced_data'):
        """
        Print load balance and communication statistics
        Similar to printing iteration counts in Poisson solver
        """
        # Gather statistics from all processes
        all_local_queries = self.comm.gather(self.local_queries, root=0)
        all_boundary_queries = self.comm.gather(self.boundary_queries, root=0)
        all_training_counts = self.comm.gather(len(self.local_training_data), root=0)
        
        if self.rank == 0:
            print("\n" + "="*60)
            print("PARALLEL KNN STATISTICS")
            print("="*60)
            
            total_local = sum(all_local_queries)
            total_boundary = sum(all_boundary_queries)
            total_queries = total_local + total_boundary
            
            if total_queries > 0:
                print(f"\nQuery Distribution:")
                print(f"  Total queries:     {total_queries}")
                print(f"  Local queries:     {total_local:4d} ({100*total_local/total_queries:5.1f}%) - no communication needed")
                print(f"  Boundary queries:  {total_boundary:4d} ({100*total_boundary/total_queries:5.1f}%) - MPI communication used")
            
            print(f"\nLoad Balance:")
            min_points = min(all_training_counts)
            max_points = max(all_training_counts)
            avg_points = np.mean(all_training_counts)
            imbalance = max_points / min_points if min_points > 0 else float('inf')
            
            print(f"  Min training points: {min_points:4d}")
            print(f"  Max training points: {max_points:4d}")
            print(f"  Avg training points: {avg_points:6.1f}")
            print(f"  Imbalance ratio:     {imbalance:6.2f}")
            
            if imbalance > 2.0:
                print(f"\n  ⚠ High load imbalance detected (ratio > 2.0)")
                print(f"  This is EXPECTED with clustered data distributions")
                print(f"  Report this as a finding in your analysis!")
            
            print("="*60)
            
            # Save detailed statistics to JSON
            stats = {
                'timestamp': datetime.datetime.now().isoformat(),
                'n_processes': self.size,
                'local_queries': total_local,
                'boundary_queries': total_boundary,
                'training_counts': all_training_counts,
                'min_points': min_points,
                'max_points': max_points,
                'avg_points': float(avg_points),
                'imbalance_ratio': float(imbalance)
            }
            
            try:
                with open(f'{out_dir}/parallel_knn_statistics.json', 'r') as f:
                    all_stats = json.load(f)
            except FileNotFoundError:
                all_stats = []
            
            all_stats.append(stats)
            
            with open(f'{out_dir}/parallel_knn_statistics.json', 'w') as f:
                json.dump(all_stats, f, indent=2)
            
            print(f"✓ Statistics saved to: {out_dir}/parallel_knn_statistics.json")


def main(out_dir='/home/markusbredberg/Scripts/parallelised_knn_classifier/produced_data',
         weak_scaling=False,
         balanced_partitioning=False):
    """
    Main function - Load data and run parallel KNN
    
    Parameters:
    -----------
    out_dir : str
        Output directory
    weak_scaling : bool
        If True, use weak scaling test data (different test set size per process count)
        If False, use strong scaling (same test set size for all)
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("="*60)
        print("PARALLEL KNN WITH SPATIAL DECOMPOSITION")
        print("="*60)
        print(f"Number of MPI processes: {size}")
        if weak_scaling:
            print("Mode: WEAK SCALING (problem size scales with processes)")
        else:
            print("Mode: STRONG SCALING (fixed problem size)")
    
    # Initialize KNN
    k = 5
    n_dims = 512
    # Use command-line flag to control partitioning (defaults to uniform/False)
    balanced_partitioning = args.balanced if hasattr(args, 'balanced') else False
    knn = ParallelKNN(k=k, n_dims=n_dims, balanced_partitioning=balanced_partitioning, comm=comm)

    if rank == 0:
        print(f"Partitioning mode: {'DATA-DRIVEN (equal points)' if balanced_partitioning else 'UNIFORM SPATIAL'}")
    
    # Load training data
    if rank == 0:
        try:
            if weak_scaling:
                # For weak scaling, training data is in weak_scaling subdirectory
                data_path = f'{out_dir}/weak_scaling/train_embeddings.npz'
            else:
                data_path = f'{out_dir}/strong_scaling/train_embeddings.npz'
            
            data = np.load(data_path)
            embeddings = data['embeddings']
            labels = data['labels']
        except FileNotFoundError:
            print(f"\n✗ ERROR: Training data not found!")
            if weak_scaling:
                print("Run: python generate_weak_scaling_data.py")
            else:
                print("Run: python test_full_pipeline.py or python step1_extract_features.py")
            comm.Abort(1)
    else:
        embeddings = None
        labels = None
    
    # Distribute data across processes
    knn.load_and_distribute_data(embeddings, labels)
    
    # Exchange ghost regions
    knn.exchange_ghost_regions()
    
    # Load test queries
    if rank == 0:
        try:
            if weak_scaling:
                # Load test set sized for this number of processes
                test_path = f'{out_dir}/weak_scaling/test_embeddings_{size}procs.npz'
            else:
                test_path = f'{out_dir}/strong_scaling/test_embeddings.npz'
            
            test_data = np.load(test_path)
            test_queries = test_data['embeddings']
            test_labels = test_data['labels']
            print(f"\nLoaded {len(test_queries)} test queries")
            if weak_scaling:
                print(f"  ({len(test_queries)//size} queries per process)")
        except FileNotFoundError:
            print(f"\n✗ ERROR: Test data not found!")
            if weak_scaling:
                print(f"Run: python generate_weak_scaling_data.py")
            else:
                print("Run: python test_full_pipeline.py")
            comm.Abort(1)
    else:
        test_queries = None
        test_labels = None
    
    # Broadcast test data to all processes
    test_queries = comm.bcast(test_queries, root=0)
    test_labels = comm.bcast(test_labels, root=0)
    
    # Each process classifies queries in its domain
    my_query_mask = (test_queries[:, 0] >= knn.my_min_bound) & (test_queries[:, 0] <= knn.my_max_bound)
    my_queries = test_queries[my_query_mask]
    my_true_labels = test_labels[my_query_mask]
    
    if rank == 0:
        print(f"\nStarting classification...")
    
    # Classify
    if len(my_queries) > 0:
        my_predictions, my_accuracy, my_time = knn.classify_batch(my_queries, my_true_labels)
        my_correct = np.sum(my_predictions == my_true_labels)
    else:
        my_predictions = np.array([])
        my_correct = 0
        my_time = 0
        my_accuracy = 0
    
    # Gather results INCLUDING PREDICTIONS FOR VISUALIZATION
    all_correct = comm.reduce(my_correct, op=MPI.SUM, root=0)
    all_total = comm.reduce(len(my_queries), op=MPI.SUM, root=0)
    max_time = comm.reduce(my_time, op=MPI.MAX, root=0)
    
    # =====================================================
    # Gather all predictions from all processes
    # =====================================================
    all_predictions_list = comm.gather(my_predictions, root=0)
    all_query_indices_list = comm.gather(np.where(my_query_mask)[0], root=0)
    
    if rank == 0:
        overall_accuracy = all_correct / all_total if all_total > 0 else 0
        print(f"\nClassification complete!")
        print(f"Time: {max_time:.4f} seconds")
        print(f"Accuracy: {overall_accuracy*100:.2f}%")
        
        # =====================================================
        # Reconstruct full prediction array and save
        # =====================================================
        full_predictions = np.zeros(len(test_queries), dtype=int)
        for preds, indices in zip(all_predictions_list, all_query_indices_list):
            if len(preds) > 0:
                full_predictions[indices] = preds
        
        # Save predictions along with test data for visualization
        if weak_scaling:
            pred_file = f'{out_dir}/weak_scaling/predictions_{size}procs.npz'
        else:
            pred_file = f'{out_dir}/strong_scaling/predictions_{size}procs.npz'
        
        Path(pred_file).parent.mkdir(parents=True, exist_ok=True)
        np.savez(pred_file,
                 embeddings=test_queries,
                 true_labels=test_labels,
                 predictions=full_predictions)
        print(f"✓ Predictions saved to: {pred_file}")
        
        # Save results to file
        results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'n_processes': size,
            'classification_time': max_time,
            'accuracy': overall_accuracy,
            'total_queries': all_total,
            'queries_per_process': all_total // size,
            'k': k,
            'n_dims': n_dims,
            'training_points': len(embeddings),
            'test_points': all_total,
            'weak_scaling': weak_scaling,
            'balanced_partitioning': balanced_partitioning
        }
        
        # Choose output file based on scaling mode
        if weak_scaling:
            results_file = f'{out_dir}/weak_scaling/parallel_knn_results.json'
        else:
            results_file = f'{out_dir}/strong_scaling/parallel_knn_results.json'
        
        # Append to results file
        try:
            with open(results_file, 'r') as f:
                all_results = json.load(f)
        except FileNotFoundError:
            all_results = []
        
        all_results.append(results)
        
        # Ensure directory exists
        Path(results_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"✓ Results saved to: {results_file}")
    
    # Print statistics
    if weak_scaling:
        knn.print_statistics(out_dir=f'{out_dir}/weak_scaling')
    else:
        knn.print_statistics(out_dir=f'{out_dir}/strong_scaling')


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Parallel KNN Classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Strong scaling with uniform partitioning (default)
  mpirun -np 4 python step2_parallel_knn.py
  
  # Strong scaling with balanced partitioning
  mpirun -np 4 python step2_parallel_knn.py --balanced
  
  # Weak scaling with uniform partitioning
  mpirun -np 4 python step2_parallel_knn.py --weak-scaling
  
  # Weak scaling with balanced partitioning
  mpirun -np 4 python step2_parallel_knn.py --weak-scaling --balanced
        """
    )
    
    parser.add_argument('--weak-scaling', action='store_true',
                       help='Use weak scaling mode (problem size scales with processes)')
    parser.add_argument('--balanced', action='store_true',
                       help='Use balanced (data-driven) partitioning instead of uniform spatial partitioning')
    parser.add_argument('--out-dir', type=str, 
                       default='/home/markusbredberg/Scripts/parallelised_knn_classifier/produced_data',
                       help='Output directory')
    
    args = parser.parse_args()
    
    main(out_dir=args.out_dir, 
         weak_scaling=args.weak_scaling,
         balanced_partitioning=args.balanced)