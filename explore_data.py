#!/usr/bin/env python3
"""
Data Exploration Script
Visualize the embeddings and understand the latent space
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def load_and_inspect_data():
    """
    Load the .npz files and show what's inside
    """
    print("="*60)
    print("DATA EXPLORATION")
    print("="*60)
    
    # Load training data
    print("\n1. TRAINING DATA (radio_galaxy_embeddings.npz)")
    print("-"*60)
    try:
        train_data = np.load('radio_galaxy_embeddings.npz')
        
        print("Available keys:", list(train_data.keys()))
        
        embeddings = train_data['embeddings']
        labels = train_data['labels']
        filenames = train_data['filenames']
        
        print(f"\nEmbeddings shape: {embeddings.shape}")
        print(f"  - Number of samples: {embeddings.shape[0]}")
        print(f"  - Embedding dimension: {embeddings.shape[1]}")
        
        print(f"\nLabels shape: {labels.shape}")
        print(f"Label distribution: {np.bincount(labels)}")
        print(f"Unique labels: {np.unique(labels)}")
        
        print(f"\nFirst dimension statistics:")
        print(f"  Min: {embeddings[:, 0].min():.4f}")
        print(f"  Max: {embeddings[:, 0].max():.4f}")
        print(f"  Mean: {embeddings[:, 0].mean():.4f}")
        print(f"  Std: {embeddings[:, 0].std():.4f}")
        
        print(f"\nFirst 5 filenames: {filenames[:5]}")
        
    except FileNotFoundError:
        print("✗ File not found! Run test_full_pipeline.py first")
        return None, None, None, None
    
    # Load test data
    print("\n2. TEST DATA (test_embeddings.npz)")
    print("-"*60)
    try:
        test_data = np.load('test_embeddings.npz')
        
        test_embeddings = test_data['embeddings']
        test_labels = test_data['labels']
        
        print(f"Test embeddings shape: {test_embeddings.shape}")
        print(f"Test labels shape: {test_labels.shape}")
        print(f"Test label distribution: {np.bincount(test_labels)}")
        
    except FileNotFoundError:
        print("✗ File not found!")
        return None, None, None, None
    
    return embeddings, labels, test_embeddings, test_labels


def visualize_1d_distribution(embeddings, labels, out_dir='/home/markusbredberg/Scripts/parallelised_knn_classifier/figures'):
    """
    Visualize distribution along first dimension (what's used for spatial decomposition)
    """
    print("\n3. VISUALIZING 1D DISTRIBUTION")
    print("-"*60)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Histogram of first dimension
    axes[0].hist(embeddings[:, 0], bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('First Embedding Dimension')
    axes[0].set_ylabel('Number of Points')
    axes[0].set_title('Distribution Along First Dimension (Used for Spatial Decomposition)')
    axes[0].grid(True, alpha=0.3)
    
    # Add vertical lines showing how data would be split across processes
    for n_procs in [2, 4, 8]:
        min_val = embeddings[:, 0].min()
        max_val = embeddings[:, 0].max()
        boundaries = np.linspace(min_val, max_val, n_procs + 1)
        
        # Count points in each domain
        counts = []
        for i in range(n_procs):
            if i < n_procs - 1:
                mask = (embeddings[:, 0] >= boundaries[i]) & (embeddings[:, 0] < boundaries[i+1])
            else:
                mask = (embeddings[:, 0] >= boundaries[i]) & (embeddings[:, 0] <= boundaries[i+1])
            counts.append(mask.sum())
        
        print(f"\n{n_procs} processes:")
        print(f"  Domain boundaries: {boundaries}")
        print(f"  Points per process: {counts}")
        print(f"  Load imbalance ratio: {max(counts)/min(counts):.2f}")
    
    # Scatter plot colored by label
    axes[1].scatter(embeddings[:, 0], embeddings[:, 1], 
                    c=labels, cmap='viridis', alpha=0.6, s=20)
    axes[1].set_xlabel('First Embedding Dimension')
    axes[1].set_ylabel('Second Embedding Dimension')
    axes[1].set_title('First Two Dimensions (Colored by Class)')
    axes[1].colorbar = plt.colorbar(axes[1].collections[0], ax=axes[1], label='Class')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/embedding_1d_distribution.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: embedding_1d_distribution.png")


def visualize_2d_projection(embeddings, labels, out_dir='/home/markusbredberg/Scripts/parallelised_knn_classifier/figures'):
    """
    Project high-dimensional embeddings to 2D for visualization
    """
    print("\n4. VISUALIZING 2D PROJECTIONS")
    print("-"*60)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # PCA projection
    print("Computing PCA...")
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)
    
    axes[0].scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                    c=labels, cmap='viridis', alpha=0.6, s=30)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    axes[0].set_title('PCA Projection (Linear)')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(axes[0].collections[0], ax=axes[0], label='Class')
    
    # t-SNE projection (only if reasonable number of points)
    if len(embeddings) <= 2000:
        print("Computing t-SNE (this may take a moment)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_tsne = tsne.fit_transform(embeddings)
        
        axes[1].scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], 
                        c=labels, cmap='viridis', alpha=0.6, s=30)
        axes[1].set_xlabel('t-SNE Dimension 1')
        axes[1].set_ylabel('t-SNE Dimension 2')
        axes[1].set_title('t-SNE Projection (Non-linear)')
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(axes[1].collections[0], ax=axes[1], label='Class')
    else:
        axes[1].text(0.5, 0.5, 'Too many points for t-SNE\n(showing PCA only)', 
                     ha='center', va='center', fontsize=14)
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/embedding_2d_projections.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: embedding_2d_projections.png")


def visualize_domain_decomposition(embeddings, labels):
    """
    Show how data is partitioned across processes
    """
    print("\n5. VISUALIZING DOMAIN DECOMPOSITION")
    print("-"*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Spatial Decomposition Across Different Process Counts', fontsize=14, y=0.995)
    
    process_counts = [1, 2, 4, 8]
    
    for idx, n_procs in enumerate(process_counts):
        ax = axes[idx // 2, idx % 2]
        
        # Compute domain boundaries
        min_val = embeddings[:, 0].min()
        max_val = embeddings[:, 0].max()
        boundaries = np.linspace(min_val, max_val, n_procs + 1)
        
        # Assign each point to a process
        process_ids = np.digitize(embeddings[:, 0], boundaries[1:-1])
        
        # Plot with different colors for each process
        scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], 
                           c=process_ids, cmap='tab10', alpha=0.6, s=20)
        
        # Draw vertical lines for domain boundaries
        for boundary in boundaries[1:-1]:
            ax.axvline(boundary, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # Count points per process
        counts = [np.sum(process_ids == i) for i in range(n_procs)]
        imbalance = max(counts) / min(counts) if min(counts) > 0 else float('inf')
        
        ax.set_xlabel('First Dimension (Decomposition Axis)')
        ax.set_ylabel('Second Dimension')
        ax.set_title(f'{n_procs} Process{"es" if n_procs > 1 else ""}\n'
                     f'Points per process: {counts}\n'
                     f'Imbalance ratio: {imbalance:.2f}')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Process ID')
    
    plt.tight_layout()
    plt.savefig('domain_decomposition.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: domain_decomposition.png")


def main():
    """
    Main exploration function
    """
    # Load and inspect data
    embeddings, labels, test_embeddings, test_labels = load_and_inspect_data()
    
    if embeddings is None:
        print("\n✗ Cannot proceed without data files")
        print("Run: python test_full_pipeline.py")
        return
    
    # Create visualizations
    visualize_1d_distribution(embeddings, labels)
    visualize_2d_projection(embeddings, labels)
    visualize_domain_decomposition(embeddings, labels)
    
    print("\n" + "="*60)
    print("EXPLORATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  1. embedding_1d_distribution.png - Shows data distribution")
    print("  2. embedding_2d_projections.png - PCA and t-SNE views")
    print("  3. domain_decomposition.png - How MPI partitions data")
    print("\nKey insight: Load imbalance comes from non-uniform data distribution!")


if __name__ == "__main__":
    main()