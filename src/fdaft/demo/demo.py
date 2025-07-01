import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import Dict, Optional
from models.fdaft import FDAFT
from utils.visualization import FDAFTVisualizer

def create_sample_planetary_images():
    """Create a pair of sample planetary images for demonstration"""
    np.random.seed(42)
    size = (512, 512)
    
    # Create base terrain
    x, y = np.meshgrid(np.linspace(0, 10, size[1]), np.linspace(0, 10, size[0]))
    
    # Image 1
    terrain1 = (np.sin(x) * np.cos(y) + 
               0.5 * np.sin(2*x) * np.cos(3*y) + 
               0.3 * np.sin(5*x) * np.cos(2*y))
    noise1 = np.random.normal(0, 0.1, size)
    image1 = terrain1 + noise1
    
    # Add craters to image 1
    for _ in range(3):
        cx, cy = np.random.randint(50, size[0]-50, 2)
        radius = np.random.randint(20, 40)
        y_coords, x_coords = np.ogrid[:size[0], :size[1]]
        mask = (x_coords - cx)**2 + (y_coords - cy)**2 <= radius**2
        image1[mask] *= 0.6
    
    # Image 2 (slightly transformed version of image 1)
    # Apply rotation and translation
    center = (size[1]//2, size[0]//2)
    angle = 15  # degrees
    scale = 0.9
    
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += 30  # translation x
    M[1, 2] += 20  # translation y
    
    image2 = cv2.warpAffine(image1, M, (size[1], size[0]))
    
    # Add some different noise and lighting
    noise2 = np.random.normal(0, 0.08, size)
    image2 = image2 + noise2 + 0.1  # brightness change
    
    # Normalize both images
    image1 = (image1 - image1.min()) / (image1.max() - image1.min())
    image2 = (image2 - image2.min()) / (image2.max() - image2.min())
    
    return (image1 * 255).astype(np.uint8), (image2 * 255).astype(np.uint8)


def demo_fdaft_complete():
    """Complete demonstration of FDAFT pipeline"""
    print("FDAFT Complete Pipeline Demonstration")
    print("=" * 50)
    
    # Create sample images
    print("Creating sample planetary images...")
    image1, image2 = create_sample_planetary_images()
    
    # Initialize FDAFT
    fdaft = FDAFT(
        num_layers=3,
        sigma_0=1.0,
        descriptor_radius=24,  # Smaller for demo
        max_keypoints=500,
        nms_radius=5
    )
    
    # Perform complete matching
    results = fdaft.match_images(image1, image2)
    
    # Visualize results
    visualizer = FDAFTVisualizer()
    
    print("\nVisualizing results...")
    visualizer.plot_matching_results(results, image1, image2)
    
    # Print summary
    print("\nMatching Summary:")
    print(f"Corner features: {len(results['corner_points1'])} -> {len(results['corner_points2'])}")
    print(f"Blob features: {len(results['blob_points1'])} -> {len(results['blob_points2'])}")
    print(f"Corner matches: {len(results['corner_matches'])}")
    print(f"Blob matches: {len(results['blob_matches'])}")
    print(f"Total matches: {len(results['all_matches_pts1'])}")
    print(f"Final matches (after RANSAC): {results['num_final_matches']}")
    
    if len(results['all_matches_pts1']) > 0:
        inlier_ratio = results['num_final_matches'] / len(results['all_matches_pts1'])
        print(f"Inlier ratio: {inlier_ratio:.2f}")
    
    return results, image1, image2


if __name__ == "__main__":
    # Run complete demonstration
    results, img1, img2 = demo_fdaft_complete()