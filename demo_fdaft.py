#!/usr/bin/env python3
"""
Complete FDAFT demonstration script

This script demonstrates the complete FDAFT pipeline for matching planetary remote sensing images.
It creates synthetic planetary images, extracts features, performs matching, and visualizes results.

Usage:
    python demo_fdaft.py

Features demonstrated:
- Double-frequency scale space construction
- Corner and blob feature extraction
- GLOH descriptor computation
- Feature matching and aggregation
- RANSAC-based outlier removal
- Comprehensive result visualization
"""

import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from fdaft.models.fdaft import FDAFT
    from fdaft.utils.visualization import FDAFTVisualizer
except ImportError as e:
    print(f"Error importing FDAFT modules: {e}")
    print("Please ensure FDAFT is properly installed.")
    print("Run: pip install -e . from the project root directory")
    sys.exit(1)


def create_sample_planetary_images():
    """
    Create a pair of sample planetary images for demonstration
    
    The images simulate real planetary characteristics:
    - Weak textures typical of planetary surfaces
    - Circular crater-like structures
    - Nonlinear illumination differences
    - Geometric transformations (rotation, scaling, translation)
    
    Returns:
        tuple: (image1, image2) - pair of synthetic planetary images
    """
    print("  Creating base terrain...")
    np.random.seed(42)  # For reproducible results
    size = (512, 512)
    
    # Create base terrain using multiple frequency components
    x, y = np.meshgrid(np.linspace(0, 10, size[1]), np.linspace(0, 10, size[0]))
    
    # Multi-scale terrain generation
    terrain1 = (
        np.sin(x) * np.cos(y) +                    # Large-scale features
        0.5 * np.sin(2*x) * np.cos(3*y) +         # Medium-scale features  
        0.3 * np.sin(5*x) * np.cos(2*y) +         # Small-scale features
        0.2 * np.sin(8*x) * np.cos(5*y)           # Fine details
    )
    
    # Add realistic noise for surface texture
    noise1 = np.random.normal(0, 0.1, size)
    image1 = terrain1 + noise1
    
    print("  Adding crater structures...")
    # Add crater-like circular depressions
    crater_positions = [
        (128, 150, 25),  # (center_x, center_y, radius)
        (300, 200, 35),
        (400, 400, 20),
        (150, 350, 30),
        (250, 100, 15)
    ]
    
    for cx, cy, radius in crater_positions:
        y_coords, x_coords = np.ogrid[:size[0], :size[1]]
        crater_mask = (x_coords - cx)**2 + (y_coords - cy)**2 <= radius**2
        
        # Create realistic crater profile (Gaussian depression)
        distance = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        crater_depth = np.exp(-distance**2 / (2 * (radius/2)**2)) * 0.4
        
        # Apply crater effect
        image1[crater_mask] -= crater_depth[crater_mask]
        
        # Add crater rim (slight elevation)
        rim_mask = ((x_coords - cx)**2 + (y_coords - cy)**2 <= (radius + 3)**2) & \
                   ((x_coords - cx)**2 + (y_coords - cy)**2 > radius**2)
        image1[rim_mask] += 0.1
    
    print("  Applying geometric transformation...")
    # Create second image with realistic transformation
    center = (size[1]//2, size[0]//2)
    angle = 15  # degrees - typical viewing angle difference
    scale = 0.9  # slight scale change
    
    # Apply rotation and scaling transformation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += 30  # translation x
    M[1, 2] += 20  # translation y
    
    image2 = cv2.warpAffine(image1, M, (size[1], size[0]))
    
    print("  Simulating illumination differences...")
    # Simulate illumination differences (different sun angles)
    # Create illumination gradient
    illumination_gradient_x = np.linspace(0.8, 1.2, size[1])
    illumination_gradient_y = np.linspace(1.1, 0.9, size[0])
    illumination_map = np.outer(illumination_gradient_y, illumination_gradient_x)
    
    image2 = image2 * illumination_map
    
    # Add different noise pattern and brightness offset
    noise2 = np.random.normal(0, 0.08, size)
    image2 = image2 + noise2 + 0.1  # brightness change
    
    # Normalize both images to [0, 255] range
    image1 = ((image1 - image1.min()) / (image1.max() - image1.min()) * 255).astype(np.uint8)
    image2 = ((image2 - image2.min()) / (image2.max() - image2.min()) * 255).astype(np.uint8)
    
    print("  Synthetic planetary images created successfully!")
    return image1, image2


def display_input_images(image1, image2):
    """Display the input image pair"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.imshow(image1, cmap='gray')
    ax1.set_title('Planetary Image 1\n(Reference)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2.imshow(image2, cmap='gray')
    ax2.set_title('Planetary Image 2\n(Transformed with illumination changes)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Add image properties as text
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, f'Size: {image1.shape}\nType: Synthetic planetary surface', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=props)
    ax2.text(0.02, 0.98, f'Size: {image2.shape}\nTransform: Rotation + Scale + Translation\nIllumination: Gradient + Noise', 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()


def analyze_feature_distribution(results):
    """Analyze and display feature distribution statistics"""
    print("\n" + "="*60)
    print("DETAILED FEATURE ANALYSIS")
    print("="*60)
    
    # Feature counts
    corner_count_1 = len(results['corner_points1'])
    corner_count_2 = len(results['corner_points2'])
    blob_count_1 = len(results['blob_points1'])
    blob_count_2 = len(results['blob_points2'])
    
    print(f"Feature Distribution:")
    print(f"  Image 1: {corner_count_1} corners + {blob_count_1} blobs = {corner_count_1 + blob_count_1} total")
    print(f"  Image 2: {corner_count_2} corners + {blob_count_2} blobs = {corner_count_2 + blob_count_2} total")
    
    # Matching statistics
    corner_matches = len(results['corner_matches'])
    blob_matches = len(results['blob_matches'])
    total_matches = len(results['all_matches_pts1'])
    final_matches = results['num_final_matches']
    
    print(f"\nMatching Performance:")
    print(f"  Corner matches: {corner_matches}")
    print(f"  Blob matches: {blob_matches}")
    print(f"  Total initial matches: {total_matches}")
    print(f"  Final matches (after RANSAC): {final_matches}")
    
    if total_matches > 0:
        inlier_ratio = final_matches / total_matches
        corner_contribution = corner_matches / total_matches
        blob_contribution = blob_matches / total_matches
        
        print(f"\nQuality Metrics:")
        print(f"  Inlier ratio: {inlier_ratio:.2%}")
        print(f"  Corner match contribution: {corner_contribution:.2%}")
        print(f"  Blob match contribution: {blob_contribution:.2%}")
        
        # Match rates (percentage of features that found matches)
        if corner_count_1 > 0 and corner_count_2 > 0:
            corner_match_rate = corner_matches / min(corner_count_1, corner_count_2)
            print(f"  Corner match rate: {corner_match_rate:.2%}")
            
        if blob_count_1 > 0 and blob_count_2 > 0:
            blob_match_rate = blob_matches / min(blob_count_1, blob_count_2)
            print(f"  Blob match rate: {blob_match_rate:.2%}")


def save_results(results, image1, image2, output_dir="demo_results"):
    """Save demonstration results to files"""
    try:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save images
        cv2.imwrite(os.path.join(output_dir, "input_image1.png"), image1)
        cv2.imwrite(os.path.join(output_dir, "input_image2.png"), image2)
        
        # Save feature points as text files
        np.savetxt(os.path.join(output_dir, "corner_points1.txt"), results['corner_points1'], fmt='%d')
        np.savetxt(os.path.join(output_dir, "corner_points2.txt"), results['corner_points2'], fmt='%d')
        np.savetxt(os.path.join(output_dir, "blob_points1.txt"), results['blob_points1'], fmt='%d')
        np.savetxt(os.path.join(output_dir, "blob_points2.txt"), results['blob_points2'], fmt='%d')
        
        # Save matches
        if len(results['corner_matches']) > 0:
            np.savetxt(os.path.join(output_dir, "corner_matches.txt"), results['corner_matches'], fmt='%d')
        if len(results['blob_matches']) > 0:
            np.savetxt(os.path.join(output_dir, "blob_matches.txt"), results['blob_matches'], fmt='%d')
        if len(results['filtered_matches_pts1']) > 0:
            np.savetxt(os.path.join(output_dir, "final_matches_pts1.txt"), results['filtered_matches_pts1'], fmt='%d')
            np.savetxt(os.path.join(output_dir, "final_matches_pts2.txt"), results['filtered_matches_pts2'], fmt='%d')
        
        print(f"\nResults saved to: {output_dir}/")
        
    except Exception as e:
        print(f"Warning: Could not save results: {e}")


def main():
    """Main demonstration function"""
    print("FDAFT Complete Pipeline Demonstration")
    print("=" * 50)
    print("This demo showcases the Fast Double-Channel Aggregated Feature Transform")
    print("for matching planetary remote sensing images.")
    print()
    
    # Step 1: Create sample images
    print("Step 1: Creating synthetic planetary images...")
    start_time = time.time()
    image1, image2 = create_sample_planetary_images()
    creation_time = time.time() - start_time
    print(f"  ✓ Images created in {creation_time:.2f} seconds")
    
    # Display input images
    print("\nDisplaying input images...")
    display_input_images(image1, image2)
    
    # Step 2: Initialize FDAFT
    print("\nStep 2: Initializing FDAFT model...")
    fdaft = FDAFT(
        num_layers=3,           # Scale space layers
        sigma_0=1.0,            # Initial scale parameter
        descriptor_radius=32,   # GLOH descriptor radius (reduced for demo speed)
        max_keypoints=600,      # Maximum keypoints per image
        nms_radius=5           # Non-maximum suppression radius
    )
    
    print("  ✓ FDAFT initialized with parameters:")
    print(f"    - Scale space layers: {fdaft.num_layers}")
    print(f"    - Descriptor radius: {fdaft.descriptor_radius}")
    print(f"    - Max keypoints: {fdaft.max_keypoints}")
    print(f"    - GLOH descriptor size: {fdaft.descriptor.get_descriptor_size()}")
    
    # Step 3: Perform complete matching pipeline
    print("\nStep 3: Executing FDAFT matching pipeline...")
    start_time = time.time()
    
    try:
        results = fdaft.match_images(image1, image2)
        matching_time = time.time() - start_time
        print(f"  ✓ Matching completed in {matching_time:.2f} seconds")
        
    except Exception as e:
        print(f"  ✗ Error during matching: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Analyze results
    analyze_feature_distribution(results)
    
    # Step 5: Visualize results
    print(f"\nStep 4: Visualizing results...")
    try:
        visualizer = FDAFTVisualizer()
        visualizer.plot_matching_results(results, image1, image2)
        print("  ✓ Visualization completed")
        
    except Exception as e:
        print(f"  ✗ Visualization error: {e}")
        # Continue anyway
    
    # Step 6: Save results
    print(f"\nStep 5: Saving results...")
    save_results(results, image1, image2)
    
    # Final summary
    print("\n" + "="*60)
    print("DEMONSTRATION SUMMARY")
    print("="*60)
    print(f"Processing time: {matching_time:.2f} seconds")
    print(f"Final matches found: {results['num_final_matches']}")
    
    if results['num_final_matches'] >= 10:
        print("✓ SUCCESS: FDAFT successfully matched the planetary images!")
        print("  The algorithm demonstrated robust performance on:")
        print("  - Weak surface textures")
        print("  - Geometric transformations (rotation, scale, translation)")
        print("  - Illumination differences")
        print("  - Noise and artifacts")
    else:
        print("⚠ WARNING: Limited matches found.")
        print("  This may be due to:")
        print("  - Insufficient distinctive features in synthetic images")
        print("  - Too restrictive matching parameters")
        print("  - Need for parameter tuning")
    
    print(f"\nNext steps:")
    print(f"  - Try with real planetary images")
    print(f"  - Adjust parameters for your specific use case")
    print(f"  - Run: python scripts/extract_features.py <image_directory>")
    print(f"  - Run: python scripts/evaluate.py <dataset_directory>")
    
    return True


if __name__ == "__main__":
    """Entry point for the demonstration script"""
    try:
        # Set matplotlib backend for environments without display
        try:
            import matplotlib
            matplotlib.use('TkAgg')  # Try interactive backend
        except:
            matplotlib.use('Agg')   # Fallback to non-interactive
            print("Note: Using non-interactive matplotlib backend")
        
        success = main()
        
        if success:
            print(f"\n🎉 Demo completed successfully!")
            input("Press Enter to exit...")
        else:
            print(f"\n❌ Demo failed. Please check the error messages above.")
            
    except KeyboardInterrupt:
        print(f"\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        
    sys.exit(0)