#!/usr/bin/env python3
"""
Script to extract FDAFT features from planetary images
"""

import argparse
import os
import pickle
import numpy as np
import cv2
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fdaft.models.fdaft import FDAFT
from fdaft.utils.visualization import FDAFTVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description='Extract FDAFT features from images')
    parser.add_argument('input_dir', help='Directory containing input images')
    parser.add_argument('--output_dir', default='./features', 
                       help='Directory to save extracted features')
    parser.add_argument('--config', default='configs/fdaft/planetary.py',
                       help='Configuration file')
    parser.add_argument('--max_keypoints', type=int, default=1000,
                       help='Maximum number of keypoints to extract')
    parser.add_argument('--visualize', action='store_true',
                       help='Save visualization of extracted features')
    parser.add_argument('--image_extensions', nargs='+', 
                       default=['.png', '.jpg', '.jpeg', '.tif', '.tiff'],
                       help='Image file extensions to process')
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from Python file"""
    if os.path.exists(config_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return config_module.config
    else:
        # Default configuration
        return {
            "model": {
                "num_layers": 3,
                "sigma_0": 1.0,
                "descriptor_radius": 48,
                "max_keypoints": 1000,
                "nms_radius": 5,
            }
        }

def extract_features_from_image(fdaft_model, image_path, output_dir, visualize=False):
    """Extract features from a single image"""
    print(f"Processing {image_path}...")
    
    # Load image
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None
    
    # Extract features
    corner_points, corner_desc, blob_points, blob_desc = fdaft_model.extract_features(image)
    
    # Prepare output
    features = {
        'image_path': str(image_path),
        'image_shape': image.shape,
        'corner_points': corner_points,
        'corner_descriptors': corner_desc,
        'blob_points': blob_points,
        'blob_descriptors': blob_desc,
        'num_corner_features': len(corner_points),
        'num_blob_features': len(blob_points),
        'total_features': len(corner_points) + len(blob_points)
    }
    
    # Save features
    feature_file = output_dir / f"{Path(image_path).stem}_features.pkl"
    with open(feature_file, 'wb') as f:
        pickle.dump(features, f)
    
    print(f"Extracted {features['total_features']} features "
          f"({features['num_corner_features']} corners, {features['num_blob_features']} blobs)")
    
    # Visualize if requested
    if visualize:
        visualizer = FDAFTVisualizer()
        visualizer.plot_features(image, corner_points, blob_points, 
                                title=f"Features: {Path(image_path).name}")
        
        # Save visualization
        vis_file = output_dir / f"{Path(image_path).stem}_features.png"
        plt.savefig(vis_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    return features

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    model_config = config.get('model', {})
    
    # Initialize FDAFT
    fdaft = FDAFT(
        num_layers=model_config.get('num_layers', 3),
        sigma_0=model_config.get('sigma_0', 1.0),
        descriptor_radius=model_config.get('descriptor_radius', 48),
        max_keypoints=args.max_keypoints,
        nms_radius=model_config.get('nms_radius', 5)
    )
    
    # Find all images
    input_dir = Path(args.input_dir)
    image_files = []
    for ext in args.image_extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No images found in {input_dir} with extensions {args.image_extensions}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process images
    all_features = []
    for image_path in sorted(image_files):
        try:
            features = extract_features_from_image(
                fdaft, image_path, output_dir, args.visualize
            )
            if features:
                all_features.append(features)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    # Save summary
    summary = {
        'total_images': len(all_features),
        'total_features': sum(f['total_features'] for f in all_features),
        'average_features_per_image': np.mean([f['total_features'] for f in all_features]),
        'config': config,
        'feature_files': [f['image_path'] for f in all_features]
    }
    
    summary_file = output_dir / 'extraction_summary.pkl'
    with open(summary_file, 'wb') as f:
        pickle.dump(summary, f)
    
    print(f"\nFeature extraction completed!")
    print(f"Processed {summary['total_images']} images")
    print(f"Extracted {summary['total_features']} total features")
    print(f"Average {summary['average_features_per_image']:.1f} features per image")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()