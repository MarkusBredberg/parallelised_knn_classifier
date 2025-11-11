#!/usr/bin/env python3
"""
Advanced Visualization Script
1. Show domain decomposition with ghost regions
2. Display actual input images
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

def visualize_domains_with_ghosts(embeddings, labels, n_procs=4, out_dir='/home/markusbredberg/Scripts/parallel_project/figures/'):
    """
    Visualize domain decomposition with ghost regions clearly marked
    """
    print(f"\nVisualizing {n_procs}-process decomposition with ghost regions...")
    
    # Compute domain boundaries
    min_val = embeddings[:, 0].min()
    max_val = embeddings[:, 0].max()
    global_width = max_val - min_val
    boundaries = np.linspace(min_val, max_val, n_procs + 1)
    domain_width = global_width / n_procs
    ghost_width = domain_width * 0.15  # 15% as in the code
    
    # Assign each point to a process
    process_ids = np.digitize(embeddings[:, 0], boundaries[1:-1])
    
    # Create figure - even larger
    fig, ax = plt.subplots(figsize=(20, 14))
    
    # Plot all points colored by process - larger markers
    colors = plt.cm.tab10(np.linspace(0, 1, n_procs))
    for proc_id in range(n_procs):
        mask = process_ids == proc_id
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                  c=[colors[proc_id]], alpha=0.6, s=50, 
                  label=f'Process {proc_id}', zorder=2)
    
    # Draw domain boundaries (solid red lines) - thicker
    for i, boundary in enumerate(boundaries):
        ax.axvline(boundary, color='red', linestyle='-', linewidth=4, 
                  alpha=0.8, zorder=3, label='Domain Boundary' if i == 0 else '')
    
    # Draw ghost regions (dashed lines and shaded areas) - thicker
    for proc_id in range(n_procs):
        proc_min = boundaries[proc_id]
        proc_max = boundaries[proc_id + 1]
        
        # Left ghost region
        if proc_id > 0:
            left_ghost_boundary = proc_min + ghost_width
            ax.axvline(left_ghost_boundary, color='blue', linestyle='--', 
                      linewidth=3, alpha=0.7, zorder=3,
                      label='Ghost Region Boundary' if proc_id == 1 else '')
            # Shade left ghost region
            ax.axvspan(proc_min, left_ghost_boundary, alpha=0.15, 
                      color='blue', zorder=1)
        
        # Right ghost region
        if proc_id < n_procs - 1:
            right_ghost_boundary = proc_max - ghost_width
            ax.axvline(right_ghost_boundary, color='blue', linestyle='--', 
                      linewidth=3, alpha=0.7, zorder=3)
            # Shade right ghost region
            ax.axvspan(right_ghost_boundary, proc_max, alpha=0.15, 
                      color='blue', zorder=1)
    
    # Add annotations with smart positioning to avoid overlap
    y_max = embeddings[:, 1].max()
    y_min = embeddings[:, 1].min()
    y_range = y_max - y_min
    
    # Calculate domain widths to determine if labels will overlap
    # Use alternating rows for all cases to ensure no overlap
    for proc_id in range(n_procs):
        proc_min = boundaries[proc_id]
        proc_max = boundaries[proc_id + 1]
        proc_center = (proc_min + proc_max) / 2
        
        # Count points
        mask = process_ids == proc_id
        n_points = mask.sum()
        
        # Alternate between two vertical positions
        if proc_id % 2 == 0:
            label_y = y_max + y_range * 0.18
        else:
            label_y = y_max + y_range * 0.32
        
        # Add text box with much larger font
        ax.text(proc_center, label_y, 
               f'Process {proc_id}\n{n_points} points', 
               ha='center', va='bottom', fontsize=16, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.6', facecolor=colors[proc_id], 
                        alpha=0.85, edgecolor='black', linewidth=2))
    
    # Add legend with clear explanation - much larger fonts
    legend_elements = [
        mpatches.Patch(color='red', label='Domain Boundaries (MPI process boundaries)'),
        mpatches.Patch(color='blue', alpha=0.3, label='Ghost Regions (overlapping data)'),
        plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=3, 
                   label='Ghost Region Limits')
    ]
    
    first_legend = ax.legend(handles=legend_elements, loc='upper right', 
                            fontsize=15, framealpha=0.95, edgecolor='black', 
                            fancybox=True, shadow=True, frameon=True)
    ax.add_artist(first_legend)
    
    # Add process legend - larger
    ax.legend(loc='upper left', fontsize=14, framealpha=0.95, ncol=2,
             edgecolor='black', fancybox=True, shadow=True, frameon=True)
    
    ax.set_xlabel('First Embedding Dimension (Decomposition Axis)', 
                 fontsize=20, fontweight='bold')
    ax.set_ylabel('Second Embedding Dimension', fontsize=20, fontweight='bold')
    ax.set_title(f'Domain Decomposition with Ghost Regions ({n_procs} Processes)\n'
                f'Ghost Width: {ghost_width:.3f} (15% of domain width)', 
                fontsize=22, fontweight='bold', pad=40)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1.5)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', labelsize=16)
    
    # Add more space at top for alternating labels
    y_top = y_max + y_range * 0.45
    ax.set_ylim(y_min - y_range * 0.05, y_top)
    
    plt.tight_layout()
    filename = f'{out_dir}/domain_with_ghosts_{n_procs}procs.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    
    # Print ghost region statistics
    print(f"\nGhost Region Statistics for {n_procs} processes:")
    print(f"  Domain width: {domain_width:.4f}")
    print(f"  Ghost width: {ghost_width:.4f} (15% overlap)")
    print(f"  Ghost/Domain ratio: {ghost_width/domain_width*100:.1f}%")
    
    for proc_id in range(n_procs):
        proc_min = boundaries[proc_id]
        proc_max = boundaries[proc_id + 1]
        
        # Count points in domain
        domain_mask = (embeddings[:, 0] >= proc_min) & (embeddings[:, 0] < proc_max)
        domain_points = domain_mask.sum()
        
        # Count points in left ghost
        left_ghost_points = 0
        if proc_id > 0:
            left_mask = (embeddings[:, 0] >= proc_min) & (embeddings[:, 0] < proc_min + ghost_width)
            left_ghost_points = left_mask.sum()
        
        # Count points in right ghost
        right_ghost_points = 0
        if proc_id < n_procs - 1:
            right_mask = (embeddings[:, 0] > proc_max - ghost_width) & (embeddings[:, 0] <= proc_max)
            right_ghost_points = right_mask.sum()
        
        total_ghost = left_ghost_points + right_ghost_points
        
        print(f"\n  Process {proc_id}:")
        print(f"    Domain: [{proc_min:.3f}, {proc_max:.3f}]")
        print(f"    Local points: {domain_points}")
        print(f"    Ghost points (left): {left_ghost_points}")
        print(f"    Ghost points (right): {right_ghost_points}")
        print(f"    Total with ghosts: {domain_points + total_ghost}")


def generate_synthetic_images(n_images=16):
    """
    Generate synthetic 'radio galaxy' images
    These simulate what would be input to the encoder
    """
    print(f"\nGenerating {n_images} synthetic radio galaxy images...")
    
    images = []
    image_size = 128
    
    for i in range(n_images):
        # Create blank image
        img = np.zeros((image_size, image_size))
        
        # Add different morphologies
        morph_type = i % 4
        
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
    
    return np.array(images)


def display_input_images(images, n_display=16, out_dir='/home/markusbredberg/Scripts/parallel_project/figures/'):
    """
    Display grid of input images
    """
    print(f"\nDisplaying {n_display} input images...")
    
    n_rows = 4
    n_cols = 4
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 18))
    fig.suptitle('Synthetic Radio Galaxy Images (Input to Encoder)', 
                fontsize=24, fontweight='bold', y=0.995)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(images):
            ax.imshow(images[idx], cmap='viridis', origin='lower')
            ax.set_title(f'Galaxy {idx}', fontsize=16, fontweight='bold', pad=10)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/input_images.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {out_dir}/input_images.png")


def create_detailed_ghost_diagram(out_dir='/home/markusbredberg/Scripts/parallel_project/figures/'):
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
    plt.savefig(f'{out_dir}/ghost_region_schematic.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {out_dir}/ghost_region_schematic.png")


def main():
    """
    Main visualization function
    """
    print("="*60)
    print("ADVANCED VISUALIZATION")
    print("="*60)
    
    # Load data
    try:
        data = np.load('radio_galaxy_embeddings.npz')
        embeddings = data['embeddings']
        labels = data['labels']
        print(f"\n✓ Loaded {len(embeddings)} embeddings")
    except FileNotFoundError:
        print("\n✗ Data not found! Run: python test_full_pipeline.py")
        return
    
    # 1. Create detailed ghost region schematic
    create_detailed_ghost_diagram()
    
    # 2. Visualize domain decomposition with ghosts for different process counts
    for n_procs in [2, 4, 8]:
        visualize_domains_with_ghosts(embeddings, labels, n_procs)
    
    # 3. Generate and display input images
    images = generate_synthetic_images(n_images=16)
    display_input_images(images)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  1. ghost_region_schematic.png - Conceptual diagram")
    print("  2. domain_with_ghosts_2procs.png - 2-process decomposition")
    print("  3. domain_with_ghosts_4procs.png - 4-process decomposition")
    print("  4. domain_with_ghosts_8procs.png - 8-process decomposition")
    print("  5. input_images.png - Synthetic radio galaxy images")
    print("\nThese show:")
    print("  • How ghost regions overlap between processes")
    print("  • Why boundary queries need ghost data")
    print("  • What the 'radio galaxies' look like")


if __name__ == "__main__":
    main()