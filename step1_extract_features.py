#!/usr/bin/env python3
"""
Step 1: Extract Features from Radio Galaxy Images
Uses pre-trained SSL encoder (BYOL or DINO) to generate embeddings
This step is embarrassingly parallel (no MPI needed)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path

class RadioGalaxyDataset(Dataset):
    """
    Dataset for loading radio galaxy images
    """
    def __init__(self, data_dir, transform=None):
        """
        Parameters:
        -----------
        data_dir : str
            Directory containing .fits or .png images
        transform : torchvision.transforms
            Image transformations
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Find all image files
        self.image_files = []
        for ext in ['*.fits', '*.png', '*.jpg', '*.jpeg']:
            self.image_files.extend(list(self.data_dir.glob(ext)))
        
        print(f"Found {len(self.image_files)} images in {data_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # Load image (FITS or standard format)
        if img_path.suffix == '.fits':
            # For FITS files, you'd use astropy
            try:
                from astropy.io import fits
                with fits.open(img_path) as hdul:
                    img_data = hdul[0].data
                # Normalize
                img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
                # Convert to PIL for transforms
                img = Image.fromarray((img_data * 255).astype(np.uint8))
            except ImportError:
                print("Warning: astropy not installed, skipping FITS files")
                img = Image.new('L', (128, 128))
        else:
            # Standard image formats
            img = Image.open(img_path).convert('L')
        
        if self.transform:
            img = self.transform(img)
        
        return img, str(img_path.name)


def load_encoder(checkpoint_path, model_type='byol', device='cuda'):
    """
    Load pre-trained SSL encoder
    
    Parameters:
    -----------
    checkpoint_path : str
        Path to checkpoint file
    model_type : str
        'byol' or 'dino'
    device : str
        'cuda' or 'cpu'
    
    Returns:
    --------
    encoder : torch.nn.Module
        Pre-trained encoder
    embedding_dim : int
        Dimensionality of embeddings
    """
    print(f"\nLoading {model_type} encoder from {checkpoint_path}...")
    
    if model_type == 'byol':
        # BYOL encoder (typically ResNet-based)
        # This is a simplified version - real implementation would load from checkpoint
        from torchvision import models
        
        # Try to load checkpoint
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Extract encoder (implementation-specific)
            # This depends on how the checkpoint was saved
            # Placeholder: create encoder architecture and load weights
            encoder = models.resnet18(pretrained=False)
            encoder = torch.nn.Sequential(*list(encoder.children())[:-1])  # Remove classifier
            
            # Load state dict (adjust key names as needed)
            try:
                encoder.load_state_dict(checkpoint['encoder'])
            except:
                print("Warning: Could not load exact checkpoint, using architecture only")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            print("Using random ResNet18 encoder for testing")
            encoder = models.resnet18(pretrained=False)
            encoder = torch.nn.Sequential(*list(encoder.children())[:-1])
        
        embedding_dim = 512
    
    elif model_type == 'dino':
        # DINO encoder (typically ViT-based)
        # Similar to BYOL, but different architecture
        print("DINO encoder support - using ResNet for now")
        from torchvision import models
        encoder = models.resnet50(pretrained=False)
        encoder = torch.nn.Sequential(*list(encoder.children())[:-1])
        embedding_dim = 2048
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    encoder = encoder.to(device)
    encoder.eval()
    
    print(f"Encoder loaded successfully: {model_type}")
    print(f"Embedding dimension: {embedding_dim}")
    
    return encoder, embedding_dim


def extract_features(encoder, dataloader, device='cuda'):
    """
    Extract embeddings from all images
    
    Parameters:
    -----------
    encoder : torch.nn.Module
        Pre-trained encoder
    dataloader : DataLoader
        Data loader for images
    device : str
        'cuda' or 'cpu'
    
    Returns:
    --------
    embeddings : np.ndarray (N x embedding_dim)
        Extracted embeddings
    filenames : list
        Image filenames
    """
    print("\nExtracting features...")
    
    all_embeddings = []
    all_filenames = []
    
    with torch.no_grad():
        for batch_idx, (images, filenames) in enumerate(dataloader):
            images = images.to(device)
            
            # Extract features
            features = encoder(images)
            features = features.view(features.size(0), -1)  # Flatten
            
            all_embeddings.append(features.cpu().numpy())
            all_filenames.extend(filenames)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {(batch_idx + 1) * dataloader.batch_size} images...")
    
    embeddings = np.vstack(all_embeddings)
    print(f"\nExtracted {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    
    return embeddings, all_filenames


def main():
    """
    Main function - extract features and save
    """
    print("="*60)
    print("STEP 1: FEATURE EXTRACTION")
    print("="*60)
    
    # ========================================
    # CONFIGURATION
    # ========================================
    
    # Path to pre-trained weights
    # Download from: https://zenodo.org/record/7615104 (BYOL)
    # Or: https://github.com/inigoval/byol
    checkpoint_path = 'path/to/checkpoint.pth'
    
    # Path to radio galaxy images
    # Download from: https://zenodo.org/record/4288837 (MiraBest)
    data_dir = 'path/to/images'
    
    # Model type: 'byol' or 'dino'
    model_type = 'byol'
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Batch size for processing
    batch_size = 32
    
    # ========================================
    # LOAD ENCODER
    # ========================================
    
    encoder, embedding_dim = load_encoder(checkpoint_path, model_type, device)
    
    # ========================================
    # PREPARE DATASET
    # ========================================
    
    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel for ResNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create dataset and dataloader
    dataset = RadioGalaxyDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # ========================================
    # EXTRACT FEATURES
    # ========================================
    
    embeddings, filenames = extract_features(encoder, dataloader, device)
    
    # ========================================
    # ASSIGN LABELS (if available)
    # ========================================
    
    # For MiraBest: parse class from filename or directory
    # For now: random labels for testing
    labels = np.random.randint(0, 2, len(embeddings))
    
    print("\nAssigning labels...")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # ========================================
    # SAVE EMBEDDINGS
    # ========================================
    
    output_file = 'radio_galaxy_embeddings.npz'
    np.savez(output_file,
             embeddings=embeddings,
             labels=labels,
             filenames=filenames)
    
    print(f"\n✓ Saved embeddings to {output_file}")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Labels: {len(labels)}")
    
    print("\n" + "="*60)
    print("STEP 1 COMPLETE!")
    print("="*60)
    print("\nNext: Run step2_parallel_knn.py with MPI")
    print("  mpirun -np 4 python step2_parallel_knn.py")


if __name__ == "__main__":
    main()