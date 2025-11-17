#!/usr/bin/env python3
"""
Data Exploration Script
1. Visualize the embeddings and understand the latent space
2. Show domain decomposition with ghost regions
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

def load_and_inspect_data():
    """
    Load the .npz files and show what's inside
    """
    print("="*60)
    print("DATA EXPLORATION")
    print("="*60)
    
    # Load training data
    print("\n1. TRAINING DATA (produced_data/strong_scaling/train_embeddings.npz)")
    print("-"*60)
    try:
        train_data = np.load('produced_data/strong_scaling/train_embeddings.npz')
        
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
    print("\n2. TEST DATA (produced_data/strong_scaling/test_embeddings.npz)")
    print("-"*60)
    try:
        test_data = np.load('produced_data/strong_scaling/test_embeddings.npz')
        
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
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Histogram of first dimension
    axes[0].hist(embeddings[:, 0], bins=50, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[0].set_xlabel('First Embedding Dimension', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Number of Points', fontsize=14, fontweight='bold')
    axes[0].set_title('Distribution Along First Dimension (Used for Spatial Decomposition)', 
                     fontsize=16, fontweight='bold')
    axes[0].tick_params(axis='both', labelsize=12)
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
            counts.append(int(mask.sum()))
        
        print(f"\n{n_procs} processes:")
        print(f"  Domain boundaries: {boundaries}")
        print(f"  Points per process: {counts}")
        print(f"  Load imbalance ratio: {max(counts)/min(counts) if min(counts) > 0 else float('inf'):.2f}")
    
    # Scatter plot with different markers and colors for each class
    unique_labels = np.unique(labels)
    markers = ['o', 's', '^', 'd', 'v', '>', '<', 'p', '*', 'h']  # Different shapes
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']  # Distinct colors
    
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        axes[1].scatter(embeddings[mask, 0], embeddings[mask, 1],
                       marker=markers[idx % len(markers)],
                       c=colors[idx % len(colors)],
                       s=50, alpha=0.7, edgecolors='black', linewidth=0.8,
                       label=f'Class {label}')
    
    axes[1].set_xlabel('First Embedding Dimension', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Second Embedding Dimension', fontsize=14, fontweight='bold')
    axes[1].set_title('First Two Dimensions by Class', fontsize=16, fontweight='bold')
    axes[1].legend(loc='best', fontsize=12, framealpha=0.95, edgecolor='black')
    axes[1].tick_params(axis='both', labelsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/fmap_embedding_1d_distribution.png', dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved: {out_dir}/fmap_embedding_1d_distribution.png")


def visualize_2d_projection(embeddings, labels, out_dir='/home/markusbredberg/Scripts/parallelised_knn_classifier/figures'):
    """
    Project high-dimensional embeddings to 2D for visualization
    """
    print("\n4. VISUALIZING 2D PROJECTIONS")
    print("-"*60)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Define markers and colors for classes
    unique_labels = np.unique(labels)
    markers = ['o', 's', '^', 'd', 'v', '>', '<', 'p', '*', 'h']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # PCA projection
    print("Computing PCA...")
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)
    
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        axes[0].scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1],
                       marker=markers[idx % len(markers)],
                       c=colors[idx % len(colors)],
                       s=50, alpha=0.7, edgecolors='black', linewidth=0.8,
                       label=f'Class {label}')
    
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', 
                      fontsize=13, fontweight='bold')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', 
                      fontsize=13, fontweight='bold')
    axes[0].set_title('PCA Projection (Linear)', fontsize=15, fontweight='bold')
    axes[0].legend(loc='best', fontsize=11, framealpha=0.95, edgecolor='black')
    axes[0].tick_params(axis='both', labelsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # t-SNE projection (only if reasonable number of points)
    if len(embeddings) <= 2000:
        print("Computing t-SNE (this may take a moment)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_tsne = tsne.fit_transform(embeddings)
        
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            axes[1].scatter(embeddings_tsne[mask, 0], embeddings_tsne[mask, 1],
                           marker=markers[idx % len(markers)],
                           c=colors[idx % len(colors)],
                           s=50, alpha=0.7, edgecolors='black', linewidth=0.8,
                           label=f'Class {label}')
        
        axes[1].set_xlabel('t-SNE Dimension 1', fontsize=13, fontweight='bold')
        axes[1].set_ylabel('t-SNE Dimension 2', fontsize=13, fontweight='bold')
        axes[1].set_title('t-SNE Projection (Non-linear)', fontsize=15, fontweight='bold')
        axes[1].legend(loc='best', fontsize=11, framealpha=0.95, edgecolor='black')
        axes[1].tick_params(axis='both', labelsize=11)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'Too many points for t-SNE\n(showing PCA only)', 
                     ha='center', va='center', fontsize=14)
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/fmap_embedding_2d_projections.png', dpi=200, bbox_inches='tight')
    print(f"✓ Saved: {out_dir}/fmap_embedding_2d_projections.png")


def visualize_domains_with_ghosts(embeddings, labels, n_procs=4, even_partitioning=False, 
                                  out_dir='/home/markusbredberg/Scripts/parallelised_knn_classifier/figures'):
    """
    Visualize domain decomposition with ghost regions clearly marked
    
    Parameters:
    -----------
    embeddings : np.ndarray
        Training embeddings
    labels : np.ndarray
        Training labels
    n_procs : int
        Number of processes
    even_partitioning : bool
        If True, use data-driven partitioning (equal points per process)
        If False, use uniform spatial partitioning
    """
    mode_name = "Balanced (Data-Driven)" if even_partitioning else "Uniform Spatial"
    print(f"\nVisualizing {n_procs}-process decomposition with ghost regions ({mode_name})...")
    
    # Compute domain boundaries
    min_val = embeddings[:, 0].min()
    max_val = embeddings[:, 0].max()
    global_width = max_val - min_val
    
    if even_partitioning:
        # Data-driven partitioning: equal points per process
        sort_indices = np.argsort(embeddings[:, 0])
        sorted_embeddings = embeddings[sort_indices]
        
        points_per_proc = len(embeddings) // n_procs
        boundaries = [min_val]
        
        for r in range(1, n_procs):
            boundary_idx = r * points_per_proc
            boundary_value = sorted_embeddings[boundary_idx, 0]
            boundaries.append(boundary_value)
        
        boundaries.append(max_val)
        boundaries = np.array(boundaries)
        
        print(f"  Using percentile-based boundaries for equal load distribution")
    else:
        # Uniform spatial partitioning
        boundaries = np.linspace(min_val, max_val, n_procs + 1)
        print(f"  Using uniform spatial boundaries")
    
    # Calculate domain widths (now variable for data-driven)
    domain_widths = np.diff(boundaries)
    ghost_widths = domain_widths * 0.15  # 15% as in the code
    
    # Assign each point to a process
    process_ids = np.digitize(embeddings[:, 0], boundaries[1:-1])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot all points colored by process
    colors = plt.cm.tab10(np.linspace(0, 1, n_procs))
    for proc_id in range(n_procs):
        mask = process_ids == proc_id
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                  c=[colors[proc_id]], alpha=0.6, s=50, 
                  label=f'Process {proc_id}', zorder=2)
    
    # Draw domain boundaries (solid red lines)
    for i, boundary in enumerate(boundaries):
        ax.axvline(boundary, color='red', linestyle='-', linewidth=4, 
                  alpha=0.8, zorder=3)
    
    # Draw ghost regions (dashed lines and shaded areas)
    for proc_id in range(n_procs):
        proc_min = boundaries[proc_id]
        proc_max = boundaries[proc_id + 1]
        ghost_width = ghost_widths[proc_id]
        
        # Left ghost region
        if proc_id > 0:
            left_ghost_boundary = proc_min + ghost_width
            ax.axvline(left_ghost_boundary, color='blue', linestyle='--', 
                      linewidth=3, alpha=0.7, zorder=3)
            ax.axvspan(proc_min, left_ghost_boundary, alpha=0.15, 
                      color='blue', zorder=1)
        
        # Right ghost region
        if proc_id < n_procs - 1:
            right_ghost_boundary = proc_max - ghost_width
            ax.axvline(right_ghost_boundary, color='blue', linestyle='--', 
                      linewidth=3, alpha=0.7, zorder=3)
            ax.axvspan(right_ghost_boundary, proc_max, alpha=0.15, 
                      color='blue', zorder=1)
    
    # Add annotations with smart positioning
    y_max = embeddings[:, 1].max()
    y_min = embeddings[:, 1].min()
    y_range = y_max - y_min
    
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
        
        # Add text box
        ax.text(proc_center, label_y, 
               f'Process {proc_id}\n{n_points} points', 
               ha='center', va='bottom', fontsize=16, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.6', facecolor=colors[proc_id], 
                        alpha=0.85, edgecolor='black', linewidth=2))
    
    # Calculate load imbalance
    point_counts = [int(np.sum(process_ids == i)) for i in range(n_procs)]
    min_points = min(point_counts)
    max_points = max(point_counts)
    imbalance = max_points / min_points if min_points > 0 else float('inf')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='red', label='Domain Boundaries (MPI process boundaries)'),
        mpatches.Patch(color='blue', alpha=0.3, label='Ghost Regions (overlapping data)'),
        plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=3, 
                   label='Ghost Region Boundaries')
    ]
    
    first_legend = ax.legend(handles=legend_elements, loc='lower right', 
                            fontsize=15, framealpha=0.95, edgecolor='black', 
                            fancybox=True, shadow=True, frameon=True)
    ax.add_artist(first_legend)
    
    ax.legend(loc='lower left', fontsize=14, framealpha=0.95, ncol=2,
             edgecolor='black', fancybox=True, shadow=True, frameon=True)
    
    ax.set_xlabel('First Embedding Dimension (Decomposition Axis)', 
                 fontsize=20, fontweight='bold')
    ax.set_ylabel('Second Embedding Dimension', fontsize=20, fontweight='bold')
    
    # Title includes load imbalance info
    title = (f'Domain Decomposition with Ghost Regions ({n_procs} Processes)\n'
            f'{mode_name} Partitioning - Load Imbalance Ratio: {imbalance:.2f}')
    ax.set_title(title, fontsize=22, fontweight='bold', pad=40)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1.5)
    
    ax.tick_params(axis='both', labelsize=16)
    
    # Add space at top
    y_top = y_max + y_range * 0.45
    ax.set_ylim(y_min - y_range * 0.05, y_top)
    
    plt.tight_layout()
    
    # Save with mode in filename
    mode_suffix = "balanced" if even_partitioning else "uniform"
    filename = f'{out_dir}/fmap_domain_with_ghosts_{n_procs}procs_{mode_suffix}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    
    # Print statistics
    print(f"\nLoad Balance Statistics for {n_procs} processes ({mode_name}):")
    print(f"  Points per process: {point_counts}")
    print(f"  Min: {min_points}, Max: {max_points}, Avg: {np.mean(point_counts):.1f}")
    print(f"  Load imbalance ratio: {imbalance:.2f}")
    
    if imbalance > 2.0:
        print(f"  ⚠ High imbalance! Consider using even_partitioning=True")
    else:
        print(f"  ✓ Good load balance!")


def compare_partitioning_strategies(embeddings, labels, n_procs=8,
                                    out_dir='/home/markusbredberg/Scripts/parallelised_knn_classifier/figures'):
    """
    Create side-by-side comparison of uniform vs balanced partitioning
    """
    print(f"\nCreating side-by-side comparison for {n_procs} processes...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    min_val = embeddings[:, 0].min()
    max_val = embeddings[:, 0].max()
    y_min = embeddings[:, 1].min()
    y_max = embeddings[:, 1].max()
    y_range = y_max - y_min
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_procs))
    
    # === LEFT: Uniform Spatial Partitioning ===
    boundaries_uniform = np.linspace(min_val, max_val, n_procs + 1)
    process_ids_uniform = np.digitize(embeddings[:, 0], boundaries_uniform[1:-1])
    
    for proc_id in range(n_procs):
        mask = process_ids_uniform == proc_id
        ax1.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                   c=[colors[proc_id]], alpha=0.6, s=40, zorder=2)
    
    for boundary in boundaries_uniform:
        ax1.axvline(boundary, color='red', linestyle='-', linewidth=3, alpha=0.8, zorder=3)
    
    # Calculate uniform load imbalance
    uniform_counts = [np.sum(process_ids_uniform == i) for i in range(n_procs)]
    uniform_imbalance = max(uniform_counts) / min(uniform_counts) if min(uniform_counts) > 0 else float('inf')
    
    ax1.set_xlabel('First Embedding Dimension', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Second Embedding Dimension', fontsize=16, fontweight='bold')
    ax1.set_title(f'Uniform Spatial Partitioning\nLoad Imbalance: {uniform_imbalance:.2f}', 
                 fontsize=18, fontweight='bold', color='red' if uniform_imbalance > 2 else 'black')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=14)
    
    # === RIGHT: Balanced Data-Driven Partitioning ===
    sort_indices = np.argsort(embeddings[:, 0])
    sorted_embeddings = embeddings[sort_indices]
    
    points_per_proc = len(embeddings) // n_procs
    boundaries_balanced = [min_val]
    
    for r in range(1, n_procs):
        boundary_idx = r * points_per_proc
        boundary_value = sorted_embeddings[boundary_idx, 0]
        boundaries_balanced.append(boundary_value)
    
    boundaries_balanced.append(max_val)
    boundaries_balanced = np.array(boundaries_balanced)
    
    process_ids_balanced = np.digitize(embeddings[:, 0], boundaries_balanced[1:-1])
    
    for proc_id in range(n_procs):
        mask = process_ids_balanced == proc_id
        ax2.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                   c=[colors[proc_id]], alpha=0.6, s=40, zorder=2,
                   label=f'Process {proc_id}')
    
    for boundary in boundaries_balanced:
        ax2.axvline(boundary, color='green', linestyle='-', linewidth=3, alpha=0.8, zorder=3)
    
    # Calculate balanced load imbalance
    balanced_counts = [np.sum(process_ids_balanced == i) for i in range(n_procs)]
    balanced_imbalance = max(balanced_counts) / min(balanced_counts) if min(balanced_counts) > 0 else float('inf')
    
    ax2.set_xlabel('First Embedding Dimension', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Second Embedding Dimension', fontsize=16, fontweight='bold')
    ax2.set_title(f'Balanced (Data-Driven) Partitioning\nLoad Imbalance: {balanced_imbalance:.2f}', 
                 fontsize=18, fontweight='bold', color='green')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', labelsize=14)
    ax2.legend(loc='upper left', fontsize=11, ncol=2, framealpha=0.9)
    
    # Overall title
    fig.suptitle(f'Partitioning Strategy Comparison ({n_procs} Processes)\n'
                f'Improvement: {uniform_imbalance/balanced_imbalance:.1f}x reduction in imbalance',
                fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    filename = f'{out_dir}/fmap_partitioning_comparison_{n_procs}procs.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    
    # Print comparison table
    print(f"\n{'='*70}")
    print(f"PARTITIONING COMPARISON ({n_procs} processes)")
    print(f"{'='*70}")
    print(f"{'Strategy':<25} {'Min Points':<12} {'Max Points':<12} {'Imbalance':<12}")
    print(f"{'-'*70}")
    print(f"{'Uniform Spatial':<25} {min(uniform_counts):<12} {max(uniform_counts):<12} {uniform_imbalance:<12.2f}")
    print(f"{'Balanced (Data-Driven)':<25} {min(balanced_counts):<12} {max(balanced_counts):<12} {balanced_imbalance:<12.2f}")
    print(f"{'='*70}")
    print(f"Improvement: {uniform_imbalance/balanced_imbalance:.1f}x reduction in load imbalance")
    print(f"{'='*70}\n")


def generate_synthetic_images(n_images=16):
    """
    Generate synthetic 'radio galaxy' images and return matching labels.
    Label assignment:
      0 -> FR-I (diffuse symmetric lobes)
      1 -> FR-II (bright hotspots)
      2 -> Bent/distorted
      3 -> Compact
    """
    print(f"\nGenerating {n_images} synthetic radio galaxy images...")
    
    images = []
    labels = []
    image_size = 128
    
    for i in range(n_images):
        # Create blank image
        img = np.zeros((image_size, image_size))
        
        # Add different morphologies
        morph_type = i % 4
        labels.append(morph_type)  # record label matching this generated image
        
        if morph_type == 0:
            # FR-I type: diffuse, symmetric lobes
            center_x, center_y = image_size // 2, image_size // 2
            
            # Central core
            y, x = np.ogrid[:image_size, :image_size]
            core_mask = (x - center_x)**2 + (y - center_y)**2 <= 5**2
            img[core_mask] = 1.0
            
            # Left lobe
            left_x, left_y = center_x - 30, center_y
            left_mask = ((x - left_x)**2 / 15**2 + (y - left_y)**2 / 20**2) <= 1
            img[left_mask] = 0.6 + 0.2 * np.random.randn(*img[left_mask].shape)
            
            # Right lobe
            right_x, right_y = center_x + 30, center_y
            right_mask = ((x - right_x)**2 / 15**2 + (y - right_y)**2 / 20**2) <= 1
            img[right_mask] = 0.6 + 0.2 * np.random.randn(*img[right_mask].shape)
            
        elif morph_type == 1:
            # FR-II type: bright hotspots at edges
            center_x, center_y = image_size // 2, image_size // 2
            
            # Central core
            y, x = np.ogrid[:image_size, :image_size]
            core_mask = (x - center_x)**2 + (y - center_y)**2 <= 3**2
            img[core_mask] = 0.8
            
            # Jets
            for offset in [-40, 40]:
                hotspot_x = center_x + offset
                hotspot_y = center_y
                hotspot_mask = (x - hotspot_x)**2 + (y - hotspot_y)**2 <= 8**2
                img[hotspot_mask] = 1.0
                
                # Connecting jet
                for j in range(abs(offset)):
                    jet_x = center_x + np.sign(offset) * j
                    jet_mask = (x - jet_x)**2 + (y - center_y)**2 <= 2**2
                    img[jet_mask] = 0.4
                    
        elif morph_type == 2:
            # Bent/distorted morphology
            center_x, center_y = image_size // 2, image_size // 2
            
            # Core
            y, x = np.ogrid[:image_size, :image_size]
            core_mask = (x - center_x)**2 + (y - center_y)**2 <= 4**2
            img[core_mask] = 0.9
            
            # Curved plume
            for t in np.linspace(0, np.pi, 50):
                plume_x = int(center_x + 35 * np.cos(t))
                plume_y = int(center_y + 25 * np.sin(t))
                if 0 <= plume_x < image_size and 0 <= plume_y < image_size:
                    plume_mask = (x - plume_x)**2 + (y - plume_y)**2 <= 8**2
                    img[plume_mask] = 0.5 + 0.2 * np.random.randn(*img[plume_mask].shape)
                    
        else:
            # Compact source
            center_x = image_size // 2 + np.random.randint(-10, 10)
            center_y = image_size // 2 + np.random.randint(-10, 10)
            y, x = np.ogrid[:image_size, :image_size]
            compact_mask = (x - center_x)**2 + (y - center_y)**2 <= 12**2
            img[compact_mask] = 0.8 + 0.2 * np.random.randn(*img[compact_mask].shape)
        
        # Add noise
        img += 0.05 * np.random.randn(image_size, image_size)
        img = np.clip(img, 0, 1)
        
        images.append(img)
    
    # return images and matching labels (both length n_images)
    return np.array(images), np.array(labels)


def create_detailed_ghost_diagram(out_dir='/home/markusbredberg/Scripts/parallelised_knn_classifier/figures'):
    """
    Create a detailed schematic diagram explaining ghost regions
    """
    print("\nCreating detailed ghost region schematic...")
    
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Example with 3 processes for clarity
    n_procs = 3
    proc_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    y_center = 0.5
    x_start = 0.05
    x_end = 0.95
    proc_width = (x_end - x_start) / n_procs
    ghost_width = proc_width * 0.15
    
    # Draw processes
    for i in range(n_procs):
        x_proc_start = x_start + i * proc_width
        x_proc_end = x_proc_start + proc_width
        
        # Main domain
        rect = Rectangle((x_proc_start, y_center - 0.15), proc_width, 0.3,
                         facecolor=proc_colors[i], alpha=0.4, edgecolor='black', 
                         linewidth=4)
        ax.add_patch(rect)
        
        # Process label - much larger font
        ax.text(x_proc_start + proc_width/2, y_center + 0.25, 
               f'Process {i}', ha='center', va='bottom', 
               fontsize=22, fontweight='bold')
        
        # Domain boundaries - thicker
        ax.axvline(x_proc_start, color='red', linewidth=5, linestyle='-', 
                  ymin=0.2, ymax=0.8)
        if i == n_procs - 1:
            ax.axvline(x_proc_end, color='red', linewidth=5, linestyle='-', 
                      ymin=0.2, ymax=0.8)
        
        # Ghost regions
        if i > 0:
            # Left ghost (receive from left neighbor)
            left_ghost = Rectangle((x_proc_start, y_center - 0.12), ghost_width, 0.24,
                                  facecolor='blue', alpha=0.3, edgecolor='blue', 
                                  linewidth=4, linestyle='--')
            ax.add_patch(left_ghost)
            ax.text(x_proc_start + ghost_width/2, y_center - 0.30, 
                   'Left\nGhost', ha='center', va='top', fontsize=16, 
                   color='blue', fontweight='bold')
        
        if i < n_procs - 1:
            # Right ghost (receive from right neighbor)
            right_ghost_start = x_proc_end - ghost_width
            right_ghost = Rectangle((right_ghost_start, y_center - 0.12), ghost_width, 0.24,
                                   facecolor='blue', alpha=0.3, edgecolor='blue', 
                                   linewidth=4, linestyle='--')
            ax.add_patch(right_ghost)
            ax.text(right_ghost_start + ghost_width/2, y_center - 0.30, 
                   'Right\nGhost', ha='center', va='top', fontsize=16, 
                   color='blue', fontweight='bold')
    
    # Add arrows showing ghost exchange - thicker
    arrow_y = 0.75
    for i in range(n_procs - 1):
        x_boundary = x_start + (i + 1) * proc_width
        
        # Right arrow (proc i sends to proc i+1)
        ax.annotate('', xy=(x_boundary + 0.06, arrow_y + 0.03), 
                   xytext=(x_boundary - 0.06, arrow_y + 0.03),
                   arrowprops=dict(arrowstyle='->', lw=4, color='green'))
        
        # Left arrow (proc i+1 sends to proc i)
        ax.annotate('', xy=(x_boundary - 0.06, arrow_y - 0.03), 
                   xytext=(x_boundary + 0.06, arrow_y - 0.03),
                   arrowprops=dict(arrowstyle='->', lw=4, color='orange'))
    
    ax.text(0.5, arrow_y + 0.12, 'MPI Ghost Region Exchange (sendrecv)', 
           ha='center', fontsize=20, fontweight='bold', color='darkgreen')
    
    # Add legend - larger fonts
    legend_elements = [
        mpatches.Patch(facecolor='gray', alpha=0.4, edgecolor='black', 
                      linewidth=3, label='Local Domain (owned by process)'),
        mpatches.Patch(facecolor='blue', alpha=0.3, edgecolor='blue', 
                      linewidth=3, linestyle='--', label='Ghost Regions (cached from neighbors)'),
        plt.Line2D([0], [0], color='red', linewidth=5, label='Domain Boundaries'),
        plt.Line2D([0], [0], color='green', linewidth=4, 
                  marker='>', markersize=12, label='Data sent TO neighbor'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', 
             fontsize=17, ncol=2, framealpha=0.95, edgecolor='black',
             fancybox=True, shadow=True)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Ghost Region Exchange Mechanism\n'
                '(Similar to Poisson Solver Ghost Cell Exchange)', 
                fontsize=24, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/fmap_ghost_region_schematic.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {out_dir}/fmap_ghost_region_schematic.png")


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
    create_detailed_ghost_diagram()
    visualize_1d_distribution(embeddings, labels)
    visualize_2d_projection(embeddings, labels)
    
    # Compare partitioning strategies (NEW!)
    print("\n" + "="*60)
    print("COMPARING PARTITIONING STRATEGIES")
    print("="*60)
    compare_partitioning_strategies(embeddings, labels, n_procs=8)
    
    # Visualize both partitioning modes for different process counts
    for n_procs in [2, 4, 8]:
        print(f"\n--- {n_procs} Processes ---")
        # Uniform partitioning
        visualize_domains_with_ghosts(embeddings, labels, n_procs, even_partitioning=False)
        # Balanced partitioning
        visualize_domains_with_ghosts(embeddings, labels, n_procs, even_partitioning=True)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  1. fmap_ghost_region_schematic.png - Conceptual diagram")
    print("  2. fmap_embedding_1d_distribution.png - Data distribution analysis")
    print("  3. fmap_embedding_2d_projections.png - PCA and t-SNE views")
    print("  4. fmap_partitioning_comparison_8procs.png - Side-by-side comparison")
    print("  5-10. fmap_domain_with_ghosts_*procs_uniform.png - Uniform partitioning")
    print("  11-16. fmap_domain_with_ghosts_*procs_balanced.png - Balanced partitioning")
    print("   It clearly shows the load imbalance problem and solution!")

if __name__ == "__main__":
    main()