#!/usr/bin/env python3
"""
Step 2: Parallel KNN Classifier with Spatial Decomposition
Operates on embeddings from pre-trained encoder (Step 1)
Uses spatial decomposition of embedding space with MPI communication
"""

import numpy as np
from mpi4py import MPI
import time
import json
import datetime

class ParallelKNN:
    """
    Parallel KNN classifier using spatial decomposition
    Similar structure to your Poisson solver!
    """
    def __init__(self, k=5, n_dims=512, comm=MPI.COMM_WORLD):
        """
        Initialize parallel KNN classifier
        
        Parameters:
        -----------
        k : int
            Number of nearest neighbors
        n_dims : int
            Dimensionality of embeddings (512 for BYOL, 2048 for DINO)
        comm : MPI.Comm
            MPI communicator
        """
        # Core KNN parameters
        self.k = k
        self.n_dims = n_dims
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
        Similar to distributing grid points in Poisson solver
        
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
            
            # Compute global domain boundaries based on first dimension
            global_min = embeddings[:, 0].min()
            global_max = embeddings[:, 0].max()
            global_width = global_max - global_min
            
            print(f"Global embedding space: [{global_min:.4f}, {global_max:.4f}]")
            
            # Distribute data to all processes
            for r in range(self.size):
                # Compute domain boundaries for process r
                r_min = global_min + r * global_width / self.size
                r_max = global_min + (r + 1) * global_width / self.size
                
                # Send boundaries to process r
                bounds = np.array([r_min, r_max, global_width / self.size])
                self.comm.send(bounds, dest=r, tag=10)
                
                # Select data in this domain
                if r < self.size - 1:
                    # Exclude upper bound for middle processes
                    mask = (embeddings[:, 0] >= r_min) & (embeddings[:, 0] < r_max)
                else:
                    # Include upper bound for last process
                    mask = (embeddings[:, 0] >= r_min) & (embeddings[:, 0] <= r_max)
                
                local_data = embeddings[mask]
                local_labels = labels[mask]
                
                if r == 0:
                    # Keep data for myself
                    self.local_training_data = local_data
                    self.local_training_labels = local_labels
                else:
                    # Send to process r
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
        
        print(f"Rank {self.rank}: Domain [{self.my_min_bound:.4f}, {self.my_max_bound:.4f}] "
              f"with {len(self.local_training_data)} training points")
        
        # Compute ghost width (based on local data density and k)
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
            print(f"Rank {self.rank}: Received {len(ghost_from_left_data)} points from left")
        
        if ghost_from_right_data is not None:
            ghost_data_list.append(ghost_from_right_data)
            ghost_labels_list.append(ghost_from_right_labels)
            print(f"Rank {self.rank}: Received {len(ghost_from_right_data)} points from right")
        
        if ghost_data_list:
            self.ghost_training_data = np.vstack(ghost_data_list)
            self.ghost_training_labels = np.concatenate(ghost_labels_list)
            print(f"Rank {self.rank}: Total ghost points = {len(self.ghost_training_data)}")
        else:
            self.ghost_training_data = np.empty((0, self.n_dims))
            self.ghost_training_labels = np.empty(0, dtype=int)
            print(f"Rank {self.rank}: No ghost points")
    
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
    
    def print_statistics(self, out_dir='/home/markusbredberg/Scripts/parallel_project/produced_data'):
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


def main(out_dir='/home/markusbredberg/Scripts/parallel_project/produced_data'):
    """
    Main function - Load data and run parallel KNN
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("="*60)
        print("PARALLEL KNN WITH SPATIAL DECOMPOSITION")
        print("="*60)
        print(f"Number of MPI processes: {size}")
    
    # Initialize KNN
    k = 5
    n_dims = 512
    knn = ParallelKNN(k=k, n_dims=n_dims, comm=comm)
    
    # Load training data
    if rank == 0:
        try:
            data = np.load('radio_galaxy_embeddings.npz')
            embeddings = data['embeddings']
            labels = data['labels']
        except FileNotFoundError:
            print("\n✗ ERROR: radio_galaxy_embeddings.npz not found!")
            print("Run step1_extract_features.py first, or use test_full_pipeline.py to generate synthetic data")
            comm.Abort(1)
    else:
        embeddings = None
        labels = None
    
    # Distribute data across processes
    knn.load_and_distribute_data(embeddings, labels)
    
    # Exchange ghost regions (like Poisson!)
    knn.exchange_ghost_regions()
    
    # Load test queries
    if rank == 0:
        try:
            test_data = np.load('test_embeddings.npz')
            test_queries = test_data['embeddings']
            test_labels = test_data['labels']
            print(f"\nLoaded {len(test_queries)} test queries")
        except FileNotFoundError:
            print("\n✗ ERROR: test_embeddings.npz not found!")
            print("Run test_full_pipeline.py to generate test data")
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
    
    # Gather results
    all_correct = comm.reduce(my_correct, op=MPI.SUM, root=0)
    all_total = comm.reduce(len(my_queries), op=MPI.SUM, root=0)
    max_time = comm.reduce(my_time, op=MPI.MAX, root=0)
    
    if rank == 0:
        overall_accuracy = all_correct / all_total if all_total > 0 else 0
        print(f"\nClassification complete!")
        print(f"Time: {max_time:.4f} seconds")
        print(f"Accuracy: {overall_accuracy*100:.2f}%")
        
        # Save results to file
        results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'n_processes': size,
            'classification_time': max_time,
            'accuracy': overall_accuracy,
            'total_queries': all_total,
            'k': k,
            'n_dims': n_dims,
            'training_points': len(embeddings),
            'test_points': all_total
        }
        
        # Append to results file
        try:
            with open(f'{out_dir}/parallel_knn_results.json', 'r') as f:
                all_results = json.load(f)
        except FileNotFoundError:
            all_results = []
        
        all_results.append(results)
        
        with open(f'{out_dir}/parallel_knn_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"✓ Results saved to: {out_dir}/parallel_knn_results.json")
    
    # Print statistics (this now also saves them to JSON)
    knn.print_statistics()


if __name__ == "__main__":
    main()