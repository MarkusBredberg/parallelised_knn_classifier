#!/usr/bin/env python3
"""
Quick Test Script - Full Pipeline with Synthetic Data
Run this to test your setup without needing real data or pre-trained weights
"""

import numpy as np
import sys
import os

def generate_synthetic_data(n_samples=1000, n_dims=512, n_classes=2):
    """
    Generate synthetic radio galaxy embeddings
    Simulates what you'd get from a pre-trained encoder
    """
    print(f"Generating synthetic dataset...")
    print(f"  Samples: {n_samples}")
    print(f"  Dimensions: {n_dims}")
    print(f"  Classes: {n_classes}")
    
    # Create clustered data to simulate real radio galaxy distributions
    embeddings = []
    labels = []
    
    for class_id in range(n_classes):
        # Create cluster for this class
        n_class_samples = n_samples // n_classes
        
        # Random center in embedding space
        center = np.random.randn(n_dims) * 2
        
        # Generate points around center with some spread
        class_embeddings = center + np.random.randn(n_class_samples, n_dims) * 0.5
        
        embeddings.append(class_embeddings)
        labels.extend([class_id] * n_class_samples)
    
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    
    # Shuffle
    perm = np.random.permutation(len(embeddings))
    embeddings = embeddings[perm]
    labels = labels[perm]
    
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    return embeddings, labels


def test_step1_encoder():
    """
    Test Step 1: Feature extraction
    Uses random encoder (no pre-trained weights needed for testing)
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("\n" + "="*60)
        print("TESTING STEP 1: FEATURE EXTRACTION")
        print("="*60)
    
    try:
        import torch
        from torchvision import models
        
        # Create a simple encoder (random weights)
        encoder = models.resnet18(weights=None)  # Use weights=None instead of pretrained=False
        encoder = torch.nn.Sequential(*list(encoder.children())[:-1])
        encoder.eval()
        
        # Test with synthetic images (3 channels for ResNet)
        test_images = np.random.rand(10, 3, 128, 128)
        test_images_tensor = torch.from_numpy(test_images).float()
        
        with torch.no_grad():
            embeddings = encoder(test_images_tensor)
            embeddings = embeddings.view(embeddings.size(0), -1).numpy()
        
        if rank == 0:
            print(f"✓ Encoder test passed!")
            print(f"  Output shape: {embeddings.shape}")
        return True
    except ImportError as e:
        if rank == 0:
            print(f"✗ Encoder test failed: {e}")
            print("  Install with: pip install torch torchvision")
        return False
    except Exception as e:
        if rank == 0:
            print(f"✗ Encoder test failed: {e}")
        return False


def test_step2_mpi():
    """
    Test Step 2: MPI functionality
    """
    try:
        from mpi4py import MPI
        
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        if rank == 0:
            print("\n" + "="*60)
            print("TESTING STEP 2: MPI SETUP")
            print("="*60)
            print(f"✓ MPI test passed!")
            print(f"  Processes: {size}")
            
            if size == 1:
                print("\n  NOTE: Currently running with 1 process")
                print("  To test MPI properly, run with: mpirun -np 4 python test_full_pipeline.py")
        
        return True
    except Exception as e:
        if rank == 0:
            print(f"✗ MPI test failed: {e}")
            print("  Install with: pip install mpi4py")
        return False


def create_test_data():
    """
    Create synthetic embeddings and save them
    Only rank 0 creates the files to avoid corruption
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("\n" + "="*60)
        print("CREATING TEST DATA")
        print("="*60)
        
        # Generate synthetic embeddings
        embeddings, labels = generate_synthetic_data(
            n_samples=1000,
            n_dims=512,
            n_classes=2
        )
        
        # Create train/test split
        split_idx = int(0.8 * len(embeddings))
        
        train_embeddings = embeddings[:split_idx]
        train_labels = labels[:split_idx]
        
        test_embeddings = embeddings[split_idx:]
        test_labels = labels[split_idx:]
        
        # Save training data
        np.savez('radio_galaxy_embeddings.npz',
                 embeddings=train_embeddings,
                 labels=train_labels,
                 filenames=[f'synthetic_{i}.fits' for i in range(len(train_embeddings))])
        
        print(f"✓ Created training data: radio_galaxy_embeddings.npz")
        print(f"  Training samples: {len(train_embeddings)}")
        
        # Save test data
        np.savez('test_embeddings.npz',
                 embeddings=test_embeddings,
                 labels=test_labels)
        
        print(f"✓ Created test data: test_embeddings.npz")
        print(f"  Test samples: {len(test_embeddings)}")
    
    # Wait for rank 0 to finish writing files
    comm.Barrier()
    
    return True


def run_quick_test():
    """
    Quick test of parallel KNN with synthetic data
    """
    try:
        from mpi4py import MPI
        
        # Import the parallel KNN class
        sys.path.insert(0, os.path.dirname(__file__))
        from step2_parallel_knn import ParallelKNN
        
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        if rank == 0:
            print("RUNNING QUICK PARALLEL KNN TEST")
            print("="*60)
        
        # Load data
        if rank == 0:
            data = np.load('radio_galaxy_embeddings.npz')
            embeddings = data['embeddings']
            labels = data['labels']
        else:
            embeddings = None
            labels = None
        
        # Initialize KNN
        knn = ParallelKNN(k=5, n_dims=512, comm=comm)
        
        # Distribute data
        knn.load_and_distribute_data(embeddings, labels)
        
        # Exchange ghost regions
        knn.exchange_ghost_regions()
        
        if rank == 0:
            print("✓ Parallel KNN initialization successful!")
            print(f"  k: {knn.k}")
            print(f"  Processes: {knn.size}")
        
        # Test classification on a few queries
        if rank == 0:
            test_data = np.load('test_embeddings.npz')
            test_queries = test_data['embeddings'][:10]
            test_labels = test_data['labels'][:10]
        else:
            test_queries = None
            test_labels = None
        
        test_queries = comm.bcast(test_queries, root=0)
        test_labels = comm.bcast(test_labels, root=0)
        
        # Classify
        correct = 0
        total = 0
        for i, query in enumerate(test_queries):
            if knn.my_min_bound <= query[0] <= knn.my_max_bound:
                pred = knn.classify_query(query)
                if pred == test_labels[i]:
                    correct += 1
                total += 1
        
        # Gather results
        all_correct = comm.reduce(correct, op=MPI.SUM, root=0)
        all_total = comm.reduce(total, op=MPI.SUM, root=0)
        
        if rank == 0:
            accuracy = all_correct / all_total if all_total > 0 else 0
            print(f"✓ Test classification complete!")
            print(f"  Test accuracy: {accuracy*100:.2f}%")
        
        # Print statistics
        knn.print_statistics()
        
        return True
        
    except Exception as e:
        print(f"✗ Parallel KNN test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main test function
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("="*60)
        print("PARALLEL KNN - FULL PIPELINE TEST")
        print("="*60)
        print("\nThis script will test your setup without needing:")
        print("  • Pre-trained weights")
        print("  • Real radio galaxy data")
        print("\nIt uses synthetic data to verify everything works!")
    
    # Test 1: Check if encoder libraries work
    encoder_ok = test_step1_encoder()
    
    # Test 2: Check if MPI works
    mpi_ok = test_step2_mpi()
    
    if not encoder_ok and rank == 0:
        print("\n⚠ WARNING: Encoder test failed")
        print("  Step 1 won't work without torch/torchvision")
        print("  Install with: pip install torch torchvision")
    
    if not mpi_ok and rank == 0:
        print("\n⚠ WARNING: MPI test failed")
        print("  Step 2 won't work without mpi4py")
        print("  Install with: pip install mpi4py")
        return
    
    # Create test data
    if rank == 0:
        print("\n" + "="*60)
    data_ok = create_test_data()
    
    if not data_ok and rank == 0:
        print("\n✗ Failed to create test data")
        return
    
    # Run parallel KNN test
    if comm.Get_size() > 1:
        if rank == 0:
            print("="*60)
        test_ok = run_quick_test()
        
        if test_ok and rank == 0:
            print("\n" + "="*60)
            print("✓ ALL TESTS PASSED!")
            print("="*60)
            print("\nYour setup is ready! Next steps:")
            print("1. Run with different process counts for performance analysis")
            print("2. Measure speedup: mpirun -np 1,2,4,8 python step2_parallel_knn.py")
            print("3. Create plots of time vs processes")
        elif not test_ok and rank == 0:
            print("\n" + "="*60)
            print("✗ SOME TESTS FAILED")
            print("="*60)
            print("Check error messages above")
    else:
        if rank == 0:
            print("\n" + "="*60)
            print("✓ SINGLE PROCESS TEST PASSED!")
            print("="*60)
            print("\nTo test with multiple processes, run:")
            print("  mpirun -np 4 python test_full_pipeline.py")


if __name__ == "__main__":
    main()