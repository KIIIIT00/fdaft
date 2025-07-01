import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
from torch.utils.data import Dataset

class PlanetaryImageDataset(Dataset):
    """
    Dataset class for planetary remote sensing images
    Supports Mars, Moon, and other planetary surface images
    """
    
    def __init__(self, 
                 data_dir: str,
                 image_pairs_file: Optional[str] = None,
                 transform=None):
        """
        Initialize planetary dataset
        
        Args:
            data_dir: Directory containing planetary images
            image_pairs_file: Text file with image pairs for matching
            transform: Optional transforms to apply
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Load image pairs
        if image_pairs_file:
            self.image_pairs = self._load_image_pairs(image_pairs_file)
        else:
            self.image_pairs = self._discover_image_pairs()
    
    def _load_image_pairs(self, pairs_file: str) -> List[Tuple[str, str]]:
        """Load image pairs from file"""
        pairs = []
        with open(pairs_file, 'r') as f:
            for line in f:
                img1, img2 = line.strip().split()
                pairs.append((img1, img2))
        return pairs
    
    def _discover_image_pairs(self) -> List[Tuple[str, str]]:
        """Auto-discover image pairs in directory"""
        # Simple pairing strategy - consecutive images
        image_files = sorted([f for f in os.listdir(self.data_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        
        pairs = []
        for i in range(0, len(image_files)-1, 2):
            pairs.append((image_files[i], image_files[i+1]))
        
        return pairs
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        img1_path = os.path.join(self.data_dir, self.image_pairs[idx][0])
        img2_path = os.path.join(self.data_dir, self.image_pairs[idx][1])
        
        # Load images
        image1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        
        if image1 is None or image2 is None:
            raise ValueError(f"Could not load images: {img1_path}, {img2_path}")
        
        sample = {
            'image1': image1,
            'image2': image2,
            'pair_id': idx,
            'image1_path': img1_path,
            'image2_path': img2_path
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class PlanetaryImageTransforms:
    """Common transforms for planetary images"""
    
    @staticmethod
    def normalize(sample):
        """Normalize images to [0, 1]"""
        sample['image1'] = sample['image1'].astype(np.float32) / 255.0
        sample['image2'] = sample['image2'].astype(np.float32) / 255.0
        return sample
    
    @staticmethod
    def resize(target_size=(512, 512)):
        """Resize images to target size"""
        def _resize(sample):
            sample['image1'] = cv2.resize(sample['image1'], target_size)
            sample['image2'] = cv2.resize(sample['image2'], target_size)
            return sample
        return _resize
    
    @staticmethod
    def enhance_contrast(alpha=1.2, beta=10):
        """Enhance contrast for better feature detection"""
        def _enhance(sample):
            sample['image1'] = cv2.convertScaleAbs(sample['image1'], alpha=alpha, beta=beta)
            sample['image2'] = cv2.convertScaleAbs(sample['image2'], alpha=alpha, beta=beta)
            return sample
        return _enhance