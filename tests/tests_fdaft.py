import pytest
import numpy as np
import cv2
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fdaft.models.fdaft import FDAFT
from fdaft.models.components.scale_space import DoubleFrequencyScaleSpace
from fdaft.models.components.feature_detector import FeatureDetector
from fdaft.models.components.gloh_descriptor import GLOHDescriptor

class TestFDAFT:
    """Test cases for FDAFT pipeline"""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image"""
        np.random.seed(42)
        image = np.random.rand(256, 256) * 255
        image = image.astype(np.uint8)
        
        # Add some structure (circles for craters)
        cv2.circle(image, (128, 128), 30, 100, -1)
        cv2.circle(image, (64, 64), 20, 150, -1)
        cv2.circle(image, (192, 192), 25, 80, -1)
        
        return image
    
    @pytest.fixture
    def fdaft_model(self):
        """Create FDAFT model for testing"""
        return FDAFT(
            num_layers=2,  # Smaller for faster testing
            descriptor_radius=24,
            max_keypoints=100
        )
    
    def test_scale_space_construction(self, sample_image):
        """Test double-frequency scale space construction"""
        scale_space = DoubleFrequencyScaleSpace(num_layers=2)
        
        # Test low-frequency scale space
        low_freq_space = scale_space.build_low_frequency_scale_space(sample_image)
        assert len(low_freq_space) == 2
        assert all(layer.shape == sample_image.shape for layer in low_freq_space)
        
        # Test high-frequency scale space
        high_freq_space = scale_space.build_high_frequency_scale_space(sample_image)
        assert len(high_freq_space) == 2
        assert all(layer.shape == sample_image.shape for layer in high_freq_space)
    
    def test_feature_detection(self, sample_image):
        """Test feature point detection"""
        scale_space = DoubleFrequencyScaleSpace(num_layers=2)
        detector = FeatureDetector(max_keypoints=50)
        
        # Build scale spaces
        low_freq_space = scale_space.build_low_frequency_scale_space(sample_image)
        high_freq_space = scale_space.build_high_frequency_scale_space(sample_image)
        
        # Extract features
        corner_points, corner_scores = detector.extract_corner_points(low_freq_space)
        blob_points, blob_scores = detector.extract_blob_points(high_freq_space)
        
        # Check outputs
        assert isinstance(corner_points, np.ndarray)
        assert isinstance(blob_points, np.ndarray)
        assert len(corner_scores) == len(corner_points)
        assert len(blob_scores) == len(blob_points)
        
        # Check coordinate bounds
        if len(corner_points) > 0:
            assert np.all(corner_points >= 0)
            assert np.all(corner_points[:, 0] < sample_image.shape[0])
            assert np.all(corner_points[:, 1] < sample_image.shape[1])
    
    def test_gloh_descriptor(self, sample_image):
        """Test GLOH descriptor computation"""
        descriptor = GLOHDescriptor(patch_size=49, num_radial_bins=3)
        
        # Create some test keypoints
        keypoints = np.array([[64, 64], [128, 128], [192, 192]])
        
        # Compute descriptors
        descriptors = descriptor.describe(sample_image, keypoints)
        
        # Check output shape
        expected_size = descriptor.get_descriptor_size()
        assert descriptors.shape == (len(keypoints), expected_size)
        
        # Check descriptor properties
        assert np.all(descriptors >= 0)  # Descriptors should be non-negative
        assert np.all(np.linalg.norm(descriptors, axis=1) <= 1.1)  # Should be normalized
    
    def test_feature_matching(self, fdaft_model):
        """Test feature matching between descriptors"""
        # Create dummy descriptors
        desc1 = np.random.rand(10, 100)
        desc2 = np.random.rand(15, 100)
        
        # Normalize descriptors
        desc1 = desc1 / np.linalg.norm(desc1, axis=1, keepdims=True)
        desc2 = desc2 / np.linalg.norm(desc2, axis=1, keepdims=True)
        
        # Match features
        matches = fdaft_model.match_features(desc1, desc2)
        
        # Check output format
        assert isinstance(matches, np.ndarray)
        if len(matches) > 0:
            assert matches.shape[1] == 2
            assert np.all(matches[:, 0] < len(desc1))
            assert np.all(matches[:, 1] < len(desc2))
    
    def test_complete_pipeline(self, fdaft_model, sample_image):
        """Test complete FDAFT pipeline"""
        # Create a second image (slightly modified)
        image2 = cv2.GaussianBlur(sample_image, (3, 3), 1.0)
        
        # Run complete matching pipeline
        results = fdaft_model.match_images(sample_image, image2)
        
        # Check results structure
        required_keys = [
            'corner_points1', 'corner_points2',
            'blob_points1', 'blob_points2',
            'corner_matches', 'blob_matches',
            'num_final_matches'
        ]
        
        for key in required_keys:
            assert key in results
        
        # Check data types
        assert isinstance(results['num_final_matches'], int)
        assert results['num_final_matches'] >= 0
    
    def test_ransac_filtering(self, fdaft_model):
        """Test RANSAC-based match filtering"""
        # Create synthetic matched points with some outliers
        n_inliers = 20
        n_outliers = 5
        
        # Generate inlier points (with small transformation)
        points1_inliers = np.random.rand(n_inliers, 2) * 200
        
        # Apply simple transformation (rotation + translation)
        angle = np.radians(15)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        t = np.array([10, 5])
        
        points2_inliers = (points1_inliers @ R.T) + t
        
        # Add outliers
        points1_outliers = np.random.rand(n_outliers, 2) * 200
        points2_outliers = np.random.rand(n_outliers, 2) * 200
        
        # Combine
        points1 = np.vstack([points1_inliers, points1_outliers])
        points2 = np.vstack([points2_inliers, points2_outliers])
        
        # Test RANSAC filtering
        filtered_pts1, filtered_pts2, inlier_mask = fdaft_model.filter_matches_with_ransac(
            points1, points2, threshold=2.0
        )
        
        # Check that most inliers are preserved
        assert len(filtered_pts1) >= n_inliers * 0.8  # At least 80% of inliers
        assert len(filtered_pts1) == len(filtered_pts2)
        assert len(inlier_mask) == len(points1)

def test_edge_confidence_map():
    """Test ECM computation"""
    scale_space = DoubleFrequencyScaleSpace()
    
    # Create test image with clear edges
    image = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (80, 80), 255, 2)  # Rectangle with edges
    
    # Compute ECM
    ecm = scale_space.compute_edge_confidence_map(image)
    
    # Check output properties
    assert ecm.shape == image.shape
    assert np.all(ecm >= 0) and np.all(ecm <= 1)
    assert np.sum(ecm) > 0  # Should detect some edges

if __name__ == "__main__":
    pytest.main([__file__])