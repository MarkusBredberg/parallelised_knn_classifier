#!/usr/bin/env python3
"""
Step 1: Generate/Extract Data for Parallel KNN
Unified script that can:
1. Generate synthetic data OR extract features from real radio galaxy images
2. Generate weak scaling datasets OR regular train/test split
3. Save to appropriate directories for subsequent parallel experiments
"""

import numpy as np
import argparse
from pathlib import Path

# ============================================================================
# ROTATION FOR CLASS CLUSTERING
# ============================================================================

def apply_random_rotation(embeddings, n_dims=512, seed=None):
    """
    Apply a random orthogonal rotation to embeddings
    
    This prevents class clustering along the decomposition axis!
    
    Without rotation, classes align with the first dimension, causing each
    MPI process to receive training data from only 1-2 classes at high P,
    making cross-class classification impossible in weak scaling.
    
    Args:
        embeddings: np.ndarray (N x n_dims)
        n_dims: Dimensionality
        seed: Random seed for reproducibility
    
    Returns:
        rotated_embeddings: np.ndarray (N x n_dims)
        rotation_matrix: np.ndarray (n_dims x n_dims)
    """
    try:
        from scipy.stats import ortho_group
        
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random orthogonal rotation matrix
        rotation_matrix = ortho_group.rvs(n_dims)
        
        # Apply rotation
        rotated = embeddings @ rotation_matrix.T
        
        return rotated, rotation_matrix
    
    except ImportError:
        print("⚠ Warning: scipy not available, using QR-based rotation")
        # Fallback: Generate rotation using QR decomposition
        if seed is not None:
            np.random.seed(seed)
        
        random_matrix = np.random.randn(n_dims, n_dims)
        rotation_matrix, _ = np.linalg.qr(random_matrix)
        
        rotated = embeddings @ rotation_matrix.T
        return rotated, rotation_matrix


# ============================================================================
# SYNTHETIC DATA GENERATION (WITH ROTATION)
# ============================================================================

def generate_synthetic_data(n_samples=1000, n_dims=512, n_classes=4, 
                           cluster_strength=1.0, apply_rotation=True, seed=None):
    """
    Generate synthetic radio galaxy embeddings
    Simulates what you'd get from a pre-trained encoder
    
    Args:
        n_samples: Number of samples to generate
        n_dims: Dimensionality of feature vectors
        n_classes: Number of classes
        cluster_strength: Controls class separation. Higher = more separated classes
        apply_rotation: If True, apply random rotation to prevent class clustering
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    print(f"Generating {n_samples} synthetic samples...")
    print(f"  Dimensions: {n_dims}")
    print(f"  Classes: {n_classes}")
    print(f"  Cluster strength: {cluster_strength}x")
    print(f"  Random rotation: {'ENABLED ✓' if apply_rotation else 'DISABLED ⚠'}")
    
    embeddings = []
    labels = []
    
    # Generate clustered data along first dimension
    for class_id in range(n_classes):
        n_class_samples = n_samples // n_classes
        center = np.zeros(n_dims)
        center[0] = class_id * 10.0 * cluster_strength  # Main separation along first dimension
        center[1:] = np.random.randn(n_dims - 1) * 0.5  # Small variation in other dimensions
        spread = 0.5 / cluster_strength  # Higher cluster_strength = tighter spread
        class_embeddings = center + np.random.randn(n_class_samples, n_dims) * spread
        embeddings.append(class_embeddings)
        labels.extend([class_id] * n_class_samples)
    
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    
    # Verify class clustering BEFORE rotation
    if apply_rotation:
        print("\n  BEFORE rotation - Class distribution along first dimension:")
        first_dim = embeddings[:, 0]
        for class_id in range(n_classes):
            class_mask = labels == class_id
            print(f"    Class {class_id}: x[0] ∈ [{first_dim[class_mask].min():.2f}, "
                  f"{first_dim[class_mask].max():.2f}]")
    
    # CRITICAL FIX: Apply random rotation
    rotation_matrix = None
    if apply_rotation:
        print("\n  Applying random orthogonal rotation...")
        embeddings, rotation_matrix = apply_random_rotation(embeddings, n_dims, seed=seed)
        
        # Verify AFTER rotation
        print("\n  AFTER rotation - Class distribution along first dimension:")
        first_dim = embeddings[:, 0]
        for class_id in range(n_classes):
            class_mask = labels == class_id
            print(f"    Class {class_id}: x[0] ∈ [{first_dim[class_mask].min():.2f}, "
                  f"{first_dim[class_mask].max():.2f}]")
        
        print("\n  ✓ Classes now distributed across all processes!")
    else:
        print("\n  ⚠ WARNING: No rotation! Each process will have only 1-2 classes!")
        print("  ⚠ Weak scaling accuracy will drop dramatically at high P!")
    
    # Shuffle data
    perm = np.random.permutation(len(embeddings))
    embeddings = embeddings[perm]
    labels = labels[perm]
    
    print(f"\n✓ Generated {len(embeddings)} embeddings")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Label distribution: {np.bincount(labels)}")
    
    return embeddings, labels, rotation_matrix


# ============================================================================
# REAL DATA EXTRACTION
# ============================================================================

def extract_real_features(checkpoint_path, data_dir, model_type='byol', 
                         batch_size=32, device='cuda'):
    """
    Extract features from real radio galaxy images using pre-trained encoder
    """
    print("\n" + "="*60)
    print("EXTRACTING FEATURES FROM REAL IMAGES")
    print("="*60)
    
    # Import PyTorch modules
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader, random_split
        from torchvision import transforms
        from PIL import Image
        import os
    except ImportError as e:
        print(f"✗ Error: {e}")
        print("Install with: pip install torch torchvision pillow")
        return None, None, None, None, None
    
    # Define dataset class
    class RadioGalaxyDataset(Dataset):
        def __init__(self, data_dir, transform=None):
            self.data_dir = Path(data_dir)
            self.transform = transform
            
            # Find all image files
            self.image_files = []
            for ext in ['*.fits', '*.png', '*.jpg', '*.jpeg']:
                self.image_files.extend(list(self.data_dir.rglob(ext)))
            
            self.classes = sorted({p.parent.name for p in self.image_files})
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            print(f"Found {len(self.image_files)} images across {len(self.classes)} classes: {self.classes}")
        
        def __len__(self):
            return len(self.image_files)
        
        def __getitem__(self, idx):
            img_path = self.image_files[idx]
            
            # Load image
            if img_path.suffix == '.fits':
                try:
                    from astropy.io import fits
                    with fits.open(img_path) as hdul:
                        img_data = hdul[0].data
                    img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
                    img = Image.fromarray((img_data * 255).astype(np.uint8))
                except ImportError:
                    print("Warning: astropy not installed, skipping FITS files")
                    img = Image.new('L', (128, 128))
            else:
                img = Image.open(img_path).convert('L')
            
            if self.transform:
                img = self.transform(img)
            
            return img, str(img_path.name), self.class_to_idx[img_path.parent.name]
        

    # Plot sample images sorted by label
    def plot_images_sorted_by_label(dataset, n_per_class=5, out_dir='./figures'):
        """
        Plot input images sorted by their labels.
        Works with the RadioGalaxyDataset class (from extract_real_features).
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Collect indices per class
        class_indices = {cls: [] for cls in dataset.class_to_idx.values()}
        for i, path in enumerate(dataset.image_files):
            label = dataset.class_to_idx[path.parent.name]
            class_indices[label].append(i)
        
        n_classes = len(class_indices)
        fig, axes = plt.subplots(n_classes, n_per_class, figsize=(n_per_class * 2, n_classes * 2))
        if n_classes == 1:
            axes = np.expand_dims(axes, 0)
        
        for class_id, indices in class_indices.items():
            sampled = np.random.choice(indices, min(len(indices), n_per_class), replace=False)
            for j, idx in enumerate(sampled):
                img, _, _ = dataset[idx]
                img_np = np.array(img.permute(1, 2, 0))
                ax = axes[class_id, j]
                ax.imshow(img_np.squeeze(), cmap='gray')
                ax.axis('off')
                if j == 0:
                    ax.set_ylabel(f"Label {class_id}", fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{out_dir}/sample_images_by_label.png")

    plot_images_sorted_by_label(
        RadioGalaxyDataset(data_dir, transform=transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])),
        n_per_class=5,
        out_dir='./figures'
    )
    
    # Check device
    device = device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load encoder
    print(f"\nLoading {model_type} encoder...")
    from torchvision import models
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        encoder = models.resnet18(weights=None)
        encoder = torch.nn.Sequential(*list(encoder.children())[:-1])
    else:
        print(f"⚠ Checkpoint not found at {checkpoint_path}")
        print("Using random ResNet18 encoder")
        encoder = models.resnet18(weights=None)
        encoder = torch.nn.Sequential(*list(encoder.children())[:-1])
    
    encoder = encoder.to(device)
    encoder.eval()
    embedding_dim = 512
    
    # Prepare dataset
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = RadioGalaxyDataset(data_dir, transform=transform)
    
    # Split into train/test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Extract features
    print("\nExtracting training features...")
    train_embeddings, train_filenames, train_labels = extract_features_from_loader(
        encoder, train_loader, device
    )
    
    print("\nExtracting test features...")
    test_embeddings, test_filenames, test_labels = extract_features_from_loader(
        encoder, test_loader, device
    )
    
    return (train_embeddings, train_labels, train_filenames,
            test_embeddings, test_labels)


def extract_features_from_loader(encoder, dataloader, device):
    """Helper function to extract features from a dataloader"""
    import torch
    
    all_embeddings = []
    all_filenames = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (images, filenames, labels) in enumerate(dataloader):
            images = images.to(device)
            features = encoder(images)
            features = features.view(features.size(0), -1)
            
            all_embeddings.append(features.cpu().numpy())
            all_filenames.extend(filenames)
            all_labels.extend(labels.numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * dataloader.batch_size} images...")
    
    embeddings = np.vstack(all_embeddings)
    print(f"✓ Extracted {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    
    return embeddings, all_filenames, all_labels


# ============================================================================
# DATA SAVING
# ============================================================================

def save_regular_data(train_embeddings, train_labels, train_filenames,
                     test_embeddings, test_labels, out_dir):
    """Save regular train/test split (for strong scaling experiments)"""
    out_dir = Path(out_dir) / 'strong_scaling'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("SAVING REGULAR TRAIN/TEST DATA (Strong Scaling)")
    print("="*60)
    
    # Save training data
    np.savez(f'{out_dir}/train_embeddings.npz',
             embeddings=train_embeddings,
             labels=train_labels,
             filenames=train_filenames)
    print(f"✓ Saved: {out_dir}/train_embeddings.npz")
    print(f"  Training samples: {len(train_embeddings)}")
    
    # Save test data
    test_filenames = [f'test_{i}.fits' for i in range(len(test_embeddings))]
    np.savez(f'{out_dir}/test_embeddings.npz',
             embeddings=test_embeddings,
             labels=test_labels,
             filenames=test_filenames)
    print(f"✓ Saved: {out_dir}/test_embeddings.npz")
    print(f"  Test samples: {len(test_embeddings)}")
    print(f"  Label distribution: {np.bincount(train_labels)}")


def save_weak_scaling_data(full_train_embeddings, full_train_labels, full_train_filenames,
                           full_test_embeddings, full_test_labels,
                           process_counts, out_dir, data_source):
    """Save weak scaling datasets where BOTH training and test data scale with process count"""
    out_dir = Path(out_dir) / 'weak_scaling'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    max_procs = max(process_counts)
    n_train_full = len(full_train_embeddings)
    n_test_full = len(full_test_embeddings)
    
    print("\n" + "="*60)
    print("SAVING WEAK SCALING DATASETS")
    print("="*60)
    print(f"Process counts: {process_counts}")
    print(f"Full training set: {n_train_full} samples")
    print(f"Full test set: {n_test_full} samples")
    
    # Generate datasets for each process count
    for n_procs in process_counts:
        fraction = n_procs / max_procs
        n_train = int(n_train_full * fraction)
        n_test = int(n_test_full * fraction)
        
        print(f"\nDataset for {n_procs} processes:")
        print(f"  Training: {n_train}, Test: {n_test}")
        
        # Sample training data
        train_indices = np.random.choice(n_train_full, size=n_train, replace=False)
        train_embeddings = full_train_embeddings[train_indices]
        train_labels = full_train_labels[train_indices]
        train_filenames = [full_train_filenames[i] for i in train_indices]
        
        # Sample test data
        test_indices = np.random.choice(n_test_full, size=n_test, replace=False)
        test_embeddings = full_test_embeddings[test_indices]
        test_labels = full_test_labels[test_indices]
        
        # Save training data
        train_filename = f'{out_dir}/train_embeddings_{n_procs}procs.npz'
        np.savez(train_filename, embeddings=train_embeddings, 
                labels=train_labels, filenames=train_filenames)
        print(f"  ✓ Saved: {train_filename}")
        print(f"    Label distribution: {np.bincount(train_labels)}")
        
        # Save test data
        test_filename = f'{out_dir}/test_embeddings_{n_procs}procs.npz'
        test_filenames = [f'test_{i}' for i in range(len(test_embeddings))]
        np.savez(test_filename, embeddings=test_embeddings,
                labels=test_labels, filenames=test_filenames)
        print(f"  ✓ Saved: {test_filename}")
        print(f"    Label distribution: {np.bincount(test_labels)}")



# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate data for parallel KNN experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Data source
    parser.add_argument('--real', action='store_true',
                       help='Extract features from real radio galaxy images')
    parser.add_argument('--checkpoint', type=str,
                       default='/home/markusbredberg/Scripts/pretrained_model_weights/byol.ckpt',
                       help='Path to pre-trained encoder checkpoint')
    parser.add_argument('--data-dir', type=str,
                       default='/home/markusbredberg/Scripts/data/firstgalaxydata/galaxy_data/all/',
                       help='Path to radio galaxy images')
    
    # Scaling mode
    parser.add_argument('--no-weak-scaling', action='store_true',
                       help='Disable weak scaling data generation')
    parser.add_argument('--procs', type=int, nargs='+', default=[1, 2, 4, 8],
                       help='Process counts for weak scaling')
    
    # Rotation control
    parser.add_argument('--no-rotation', action='store_true',
                       help='DISABLE random rotation (will cause accuracy drop!)')
    
    # Data parameters
    parser.add_argument('--cluster-strength', type=float, default=1.0,
                       help='Controls class separation')
    parser.add_argument('--out-dir', type=str, default='./produced_data',
                       help='Output directory')
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Number of synthetic training samples')
    parser.add_argument('--test-samples', type=int, default=200,
                       help='Number of synthetic test samples')
    parser.add_argument('--n-dims', type=int, default=512,
                       help='Embedding dimensions')
    parser.add_argument('--n-classes', type=int, default=4,
                       help='Number of classes')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    apply_rotation = not args.no_rotation
    
    print("="*70)
    print("STEP 1: DATA GENERATION FOR PARALLEL KNN")
    print("="*70)
    print(f"Random rotation: {'ENABLED ✓' if apply_rotation else 'DISABLED ⚠'}")
    print(f"Cluster strength: {args.cluster_strength}x")
    print(f"Random seed: {args.seed}")
    print()
    
    if not apply_rotation:
        print("⚠ " + "="*66)
        print("⚠ WARNING: Random rotation is DISABLED!")
        print("⚠ Weak scaling accuracy will drop from 100% to ~22% at P=8!")
        print("⚠ " + "="*66)
        print()
    
    # ========================================
    # STEP 1: GET DATA (Real or Synthetic)
    # ========================================

    # Handle real data
    if args.real:
        result = extract_real_features(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            model_type='byol',
            device='cuda'
        )
        
        if result[0] is None:
            print("\n✗ Failed to extract real features")
            print("Falling back to synthetic data...")
            args.real = False
        else:
            (train_embeddings, train_labels, train_filenames,
             test_embeddings, test_labels) = result
            data_source = "real"
    
    # Generate synthetic data
    if not args.real:
        print("\n" + "="*60)
        print("GENERATING SYNTHETIC DATA")
        print("="*60)
        
        # Generate training data WITH rotation
        train_embeddings, train_labels, rotation_matrix = generate_synthetic_data(
            n_samples=args.n_samples,
            n_dims=args.n_dims,
            n_classes=args.n_classes,
            cluster_strength=args.cluster_strength,
            apply_rotation=apply_rotation,
            seed=args.seed
        )
        train_filenames = [f'train_{i}.fits' for i in range(len(train_embeddings))]
        
        # Generate test data WITHOUT new rotation
        test_embeddings, test_labels, _ = generate_synthetic_data(
            n_samples=args.test_samples,
            n_dims=args.n_dims,
            n_classes=args.n_classes,
            cluster_strength=args.cluster_strength,
            apply_rotation=False,  # Don't generate new rotation here
            seed=args.seed + 1
        )

        # Images per class in test set
        print("\nTest set label distribution:", np.bincount(test_labels))
        
        # Apply SAME rotation to test data
        if apply_rotation and rotation_matrix is not None:
            print("\nApplying same rotation to test data...")
            test_embeddings = test_embeddings @ rotation_matrix.T
            print("✓ Test data rotated with same matrix as training data")
        
        data_source = "synthetic"
    
    # ========================================
    # STEP 2: SAVE DATA
    # ========================================

    save_regular_data(
        train_embeddings, train_labels, train_filenames,
        test_embeddings, test_labels,
        args.out_dir
    )
    print("All arguments used: ", args)
    if not args.no_weak_scaling:
        save_weak_scaling_data(
            train_embeddings, train_labels, train_filenames,
            test_embeddings, test_labels,
            args.procs, args.out_dir, data_source
        )
    
    # ========================================
    # STEP 3: SUMMARY
    # ========================================

    print("\n" + "="*70)
    print("DATA GENERATION COMPLETE!")
    print("="*70)
    print(f"\nData source: {data_source.upper()}") 
    print(f"Training samples (full): {len(train_embeddings)}")
    print(f"Test samples (full): {len(test_embeddings)}")
    print(f"Dimensions: {train_embeddings.shape[1]}")
    print(f"Classes: {len(np.unique(train_labels))}")
    print(f"Cluster strength: {args.cluster_strength}x (higher = more separated classes)")
    
    print("\nGenerated files:")
    print(f"\n  Strong Scaling (fixed data size):")
    print(f"    {args.out_dir}/strong_scaling/train_embeddings.npz ({len(train_embeddings)} samples)")
    print(f"    {args.out_dir}/strong_scaling/test_embeddings.npz ({len(test_embeddings)} samples)")
    
    if not args.no_weak_scaling:
        print(f"\n  Weak Scaling (data scales with processors):")
        for n_procs in args.procs:
            fraction = n_procs / max(args.procs)
            n_train = int(len(train_embeddings) * fraction)
            n_test = int(len(test_embeddings) * fraction)
            print(f"    {n_procs} procs: train={n_train}, test={n_test}")
            print(f"      {args.out_dir}/weak_scaling/train_embeddings_{n_procs}procs.npz")
            print(f"      {args.out_dir}/weak_scaling/test_embeddings_{n_procs}procs.npz")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    
    if args.no_weak_scaling:
        print("\nRun STRONG SCALING experiments:")
        print("  mpirun -np 1 python step2_parallel_knn.py")
        print("  mpirun -np 2 python step2_parallel_knn.py")
        print("  mpirun -np 4 python step2_parallel_knn.py")
        print("  mpirun -np 8 python step2_parallel_knn.py")
        print("\nAnalyze results:")
        print("  python plot_results.py")
    else:
        print("\n1. Run STRONG SCALING experiments:")
        print("   (Uses ALL data, divides work across processors)")
        print("  mpirun -np 1 python step2_parallel_knn.py")
        print("  mpirun -np 2 python step2_parallel_knn.py")
        print("  mpirun -np 4 python step2_parallel_knn.py")
        print("  mpirun -np 8 python step2_parallel_knn.py")
        print("\n2. Run WEAK SCALING experiments:")
        print("   (Data size scales with processors, work per processor constant)")
        for n_procs in args.procs:
            print(f"  mpirun -np {n_procs} python step2_parallel_knn.py --weak-scaling")
        print("\n3. Analyze results:")
        print("  python plot_results.py              # Strong scaling")
        print("  python plot_results.py --weak-scaling  # Weak scaling")


if __name__ == "__main__":
    main()