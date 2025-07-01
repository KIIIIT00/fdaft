import numpy as np
import cv2
from typing import Tuple, Dict, List
from .components.scale_space import DoubleFrequencyScaleSpace
from .components.feature_detector import FeatureDetector
from .components.gloh_descriptor import GLOHDescriptor

class FDAFT:
    """
    Fast Double-Channel Aggregated Feature Transform for Matching Planetary Remote Sensing Images
    
    Main class that orchestrates the complete FDAFT pipeline:
    1. Double-frequency scale space construction
    2. Feature point detection (corners and blobs)
    3. GLOH descriptor computation
    4. Feature aggregation and matching
    """
    
    def __init__(self, 
                 num_layers: int = 3,
                 sigma_0: float = 1.0,
                 descriptor_radius: int = 48,
                 max_keypoints: int = 1000,
                 nms_radius: int = 5):
        """
        Initialize FDAFT pipeline
        
        Args:
            num_layers: Number of layers in scale space
            sigma_0: Initial scale parameter
            descriptor_radius: Radius for GLOH descriptor
            max_keypoints: Maximum number of keypoints to extract
            nms_radius: Radius for non-maximum suppression
        """
        self.num_layers = num_layers
        self.sigma_0 = sigma_0
        self.descriptor_radius = descriptor_radius
        self.max_keypoints = max_keypoints
        self.nms_radius = nms_radius
        
        # Initialize components
        self.scale_space = DoubleFrequencyScaleSpace(num_layers, sigma_0)
        self.detector = FeatureDetector(nms_radius, max_keypoints)
        self.descriptor = GLOHDescriptor(
            patch_size=descriptor_radius * 2 + 1,
            num_radial_bins=3,
            num_angular_bins=8,
            num_orientation_bins=16
        )
    
    def extract_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract FDAFT features from input image
        
        Args:
            image: Input image (RGB or grayscale)
            
        Returns:
            corner_points: Corner point coordinates [N, 2]
            corner_descriptors: GLOH descriptors for corner points [N, D]
            blob_points: Blob point coordinates [M, 2]
            blob_descriptors: GLOH descriptors for blob points [M, D]
        """
        print("Building double-frequency scale space...")
        
        # Build scale spaces
        low_freq_space = self.scale_space.build_low_frequency_scale_space(image)
        high_freq_space = self.scale_space.build_high_frequency_scale_space(image)
        
        print("Extracting feature points...")
        
        # Extract feature points
        corner_points, corner_scores = self.detector.extract_corner_points(low_freq_space)
        blob_points, blob_scores = self.detector.extract_blob_points(high_freq_space)
        
        print(f"Extracted {len(corner_points)} corner points and {len(blob_points)} blob points")
        
        # Convert to grayscale for descriptor computation
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
        
        print("Computing GLOH descriptors...")
        
        # Compute descriptors
        corner_descriptors = self.descriptor.describe(gray_image, corner_points)
        blob_descriptors = self.descriptor.describe(gray_image, blob_points)
        
        print("Feature extraction completed!")
        
        return corner_points, corner_descriptors, blob_points, blob_descriptors
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray, 
                      ratio_threshold: float = 0.8) -> np.ndarray:
        """
        Match feature descriptors using nearest neighbor ratio test
        
        Args:
            desc1: Descriptors from first image [N, D]
            desc2: Descriptors from second image [M, D]
            ratio_threshold: Ratio threshold for Lowe's ratio test
            
        Returns:
            matches: Array of matches [K, 2] where each row is [idx1, idx2]
        """
        if len(desc1) == 0 or len(desc2) == 0:
            return np.empty((0, 2), dtype=int)
        
        # Compute pairwise distances
        distances = np.linalg.norm(desc1[:, np.newaxis] - desc2[np.newaxis, :], axis=2)
        
        matches = []
        
        for i in range(len(desc1)):
            # Find two nearest neighbors
            sorted_indices = np.argsort(distances[i])
            
            if len(sorted_indices) >= 2:
                nearest_dist = distances[i, sorted_indices[0]]
                second_nearest_dist = distances[i, sorted_indices[1]]
                
                # Lowe's ratio test
                if nearest_dist / second_nearest_dist < ratio_threshold:
                    matches.append([i, sorted_indices[0]])
        
        return np.array(matches) if matches else np.empty((0, 2), dtype=int)
    
    def aggregate_matches(self, corner_matches: np.ndarray, blob_matches: np.ndarray,
                         corner_points1: np.ndarray, corner_points2: np.ndarray,
                         blob_points1: np.ndarray, blob_points2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aggregate matches from corner and blob features
        
        Args:
            corner_matches: Matches from corner features [N, 2]
            blob_matches: Matches from blob features [M, 2]
            corner_points1, corner_points2: Corner points from both images
            blob_points1, blob_points2: Blob points from both images
            
        Returns:
            aggregated_points1: All matched points from image 1 [K, 2]
            aggregated_points2: All matched points from image 2 [K, 2]
        """
        all_points1 = []
        all_points2 = []
        
        # Add corner matches
        if len(corner_matches) > 0:
            for match in corner_matches:
                all_points1.append(corner_points1[match[0]])
                all_points2.append(corner_points2[match[1]])
        
        # Add blob matches  
        if len(blob_matches) > 0:
            for match in blob_matches:
                all_points1.append(blob_points1[match[0]])
                all_points2.append(blob_points2[match[1]])
        
        if all_points1:
            return np.array(all_points1), np.array(all_points2)
        else:
            return np.empty((0, 2)), np.empty((0, 2))
    
    def filter_matches_with_ransac(self, points1: np.ndarray, points2: np.ndarray,
                                  threshold: float = 3.0, max_iters: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter matches using RANSAC with homography estimation
        
        Args:
            points1: Points from first image [N, 2]
            points2: Points from second image [N, 2]
            threshold: RANSAC threshold in pixels
            max_iters: Maximum RANSAC iterations
            
        Returns:
            filtered_points1: Inlier points from image 1
            filtered_points2: Inlier points from image 2
            inlier_mask: Boolean mask of inliers
        """
        if len(points1) < 4:
            return points1, points2, np.ones(len(points1), dtype=bool)
        
        # Convert points to proper format for cv2.findHomography
        pts1 = points1[:, [1, 0]].astype(np.float32)  # Convert [y,x] to [x,y]
        pts2 = points2[:, [1, 0]].astype(np.float32)
        
        # Estimate homography with RANSAC
        _, inlier_mask = cv2.findHomography(
            pts1, pts2, 
            cv2.RANSAC, 
            threshold,
            maxIters=max_iters
        )
        
        if inlier_mask is None:
            return np.empty((0, 2)), np.empty((0, 2)), np.array([], dtype=bool)
        
        inlier_mask = inlier_mask.flatten().astype(bool)
        
        return points1[inlier_mask], points2[inlier_mask], inlier_mask
    
    def match_images(self, image1: np.ndarray, image2: np.ndarray) -> Dict:
        """
        Complete FDAFT matching pipeline for two images
        
        Args:
            image1: First image
            image2: Second image
            
        Returns:
            Dictionary containing matching results
        """
        print("Extracting features from first image...")
        corner_pts1, corner_desc1, blob_pts1, blob_desc1 = self.extract_features(image1)
        
        print("Extracting features from second image...")
        corner_pts2, corner_desc2, blob_pts2, blob_desc2 = self.extract_features(image2)
        
        print("Matching features...")
        
        # Match corner features
        corner_matches = self.match_features(corner_desc1, corner_desc2)
        
        # Match blob features
        blob_matches = self.match_features(blob_desc1, blob_desc2)
        
        print(f"Found {len(corner_matches)} corner matches and {len(blob_matches)} blob matches")
        
        # Aggregate matches
        all_pts1, all_pts2 = self.aggregate_matches(
            corner_matches, blob_matches,
            corner_pts1, corner_pts2, blob_pts1, blob_pts2
        )
        
        # Filter with RANSAC
        if len(all_pts1) > 0:
            filtered_pts1, filtered_pts2, inlier_mask = self.filter_matches_with_ransac(all_pts1, all_pts2)
        else:
            filtered_pts1, filtered_pts2, inlier_mask = np.empty((0, 2)), np.empty((0, 2)), np.array([], dtype=bool)
        
        print(f"Final matches after RANSAC: {len(filtered_pts1)}")
        
        return {
            'corner_points1': corner_pts1,
            'corner_points2': corner_pts2,
            'blob_points1': blob_pts1,
            'blob_points2': blob_pts2,
            'corner_matches': corner_matches,
            'blob_matches': blob_matches,
            'all_matches_pts1': all_pts1,
            'all_matches_pts2': all_pts2,
            'filtered_matches_pts1': filtered_pts1,
            'filtered_matches_pts2': filtered_pts2,
            'inlier_mask': inlier_mask,
            'num_final_matches': len(filtered_pts1)
        }