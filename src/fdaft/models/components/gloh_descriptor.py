import numpy as np
import cv2
from typing import Tuple, Optional
from scipy.spatial.distance import cdist

class GLOHDescriptor:
    """
    Gradient Location and Orientation Histogram (GLOH) Descriptor
    
    Improved version of SIFT descriptor using log-polar coordinates
    Reference: Mikolajczyk, K., & Schmid, C. (2005). A performance evaluation of local descriptors.
    """
    
    def __init__(self, 
                 patch_size: int = 41,
                 num_radial_bins: int = 3,
                 num_angular_bins: int = 8,
                 num_orientation_bins: int = 16,
                 gradient_threshold: float = 0.1,
                 lambda_ori: float = 1.5,
                 lambda_desc: float = 6.0):
        """
        Initialize GLOH descriptor parameters
        
        Args:
            patch_size: Size of the patch around keypoint (should be odd)
            num_radial_bins: Number of radial bins in log-polar grid
            num_angular_bins: Number of angular bins in log-polar grid  
            num_orientation_bins: Number of orientation bins for histogram
            gradient_threshold: Threshold for gradient magnitude
            lambda_ori: Standard deviation factor for orientation assignment
            lambda_desc: Standard deviation factor for descriptor
        """
        self.patch_size = patch_size
        self.radius = patch_size // 2
        self.num_radial_bins = num_radial_bins
        self.num_angular_bins = num_angular_bins
        self.num_orientation_bins = num_orientation_bins
        self.gradient_threshold = gradient_threshold
        self.lambda_ori = lambda_ori
        self.lambda_desc = lambda_desc
        
        # Pre-compute log-polar grid
        self._setup_log_polar_grid()
        
    def _setup_log_polar_grid(self):
        """Setup log-polar coordinate grid"""
        # Create coordinate grids
        y, x = np.mgrid[-self.radius:self.radius+1, -self.radius:self.radius+1]
        
        # Convert to polar coordinates
        rho = np.sqrt(x*x + y*y)
        theta = np.arctan2(y, x)
        
        # Log-polar transformation
        # Radial bins: log-scale from center to edge
        self.rho_bins = np.zeros_like(rho, dtype=int)
        self.theta_bins = np.zeros_like(theta, dtype=int)
        self.bin_mask = rho <= self.radius
        
        # Define radial bin boundaries (log scale)
        max_rho = self.radius
        if self.num_radial_bins == 3:
            # Standard GLOH: center circle + 2 rings
            radial_bounds = [0, max_rho/3, 2*max_rho/3, max_rho]
        else:
            # General case: log scale
            radial_bounds = np.logspace(0, np.log10(max_rho), self.num_radial_bins + 1)
            radial_bounds[0] = 0
            
        # Assign radial bins
        for i in range(self.num_radial_bins):
            mask = (rho >= radial_bounds[i]) & (rho < radial_bounds[i+1])
            self.rho_bins[mask] = i
            
        # Assign angular bins
        theta_normalized = (theta + np.pi) / (2 * np.pi)  # Normalize to [0, 1]
        self.theta_bins = (theta_normalized * self.num_angular_bins).astype(int)
        self.theta_bins = np.clip(self.theta_bins, 0, self.num_angular_bins - 1)
        
        # Special handling for center bin
        center_mask = rho < radial_bounds[1]
        self.center_mask = center_mask
        
    def compute_gradients(self, patch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradient magnitude and orientation
        
        Args:
            patch: Image patch around keypoint
            
        Returns:
            magnitude: Gradient magnitude
            orientation: Gradient orientation in radians [-π, π]
        """
        # Apply Gaussian smoothing for stability
        patch_smooth = cv2.GaussianBlur(patch.astype(np.float32), (3, 3), 0.5)
        
        # Compute gradients using Sobel operators
        grad_x = cv2.Sobel(patch_smooth, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(patch_smooth, cv2.CV_32F, 0, 1, ksize=3)
        
        # Compute magnitude and orientation
        magnitude = np.sqrt(grad_x*grad_x + grad_y*grad_y)
        orientation = np.arctan2(grad_y, grad_x)
        
        return magnitude, orientation
        
    def compute_dominant_orientation(self, magnitude: np.ndarray, 
                                   orientation: np.ndarray) -> float:
        """
        Compute dominant orientation for the keypoint
        
        Args:
            magnitude: Gradient magnitude
            orientation: Gradient orientation
            
        Returns:
            Dominant orientation in radians
        """
        # Create orientation histogram
        hist_bins = 36  # 10-degree bins
        hist, bin_edges = np.histogram(
            orientation.flatten(), 
            bins=hist_bins, 
            range=(-np.pi, np.pi),
            weights=magnitude.flatten()
        )
        
        # Smooth histogram with Gaussian
        sigma = self.lambda_ori
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        hist_smooth = cv2.filter2D(hist.reshape(1, -1).astype(np.float32), 
                                  -1, kernel.T).flatten()
        
        # Find peaks
        peak_idx = np.argmax(hist_smooth)
        
        # Parabolic interpolation for sub-bin accuracy
        if 0 < peak_idx < len(hist_smooth) - 1:
            left = hist_smooth[peak_idx - 1]
            center = hist_smooth[peak_idx]
            right = hist_smooth[peak_idx + 1]
            
            # Parabolic fit
            offset = 0.5 * (left - right) / (left - 2*center + right + 1e-6)
            peak_bin = peak_idx + offset
        else:
            peak_bin = peak_idx
            
        # Convert bin to orientation
        dominant_orientation = -np.pi + (peak_bin / hist_bins) * 2 * np.pi
        
        return dominant_orientation
        
    def compute_descriptor(self, patch: np.ndarray, 
                          dominant_orientation: float) -> np.ndarray:
        """
        Compute GLOH descriptor for the patch
        
        Args:
            patch: Image patch around keypoint
            dominant_orientation: Dominant orientation for rotation invariance
            
        Returns:
            GLOH descriptor vector
        """
        # Compute gradients
        magnitude, orientation = self.compute_gradients(patch)
        
        # Rotate orientations relative to dominant orientation
        orientation_relative = orientation - dominant_orientation
        orientation_relative = np.mod(orientation_relative + np.pi, 2*np.pi) - np.pi
        
        # Apply Gaussian weighting
        sigma = self.lambda_desc
        y, x = np.mgrid[-self.radius:self.radius+1, -self.radius:self.radius+1]
        gaussian_weight = np.exp(-(x*x + y*y) / (2 * sigma*sigma))
        
        # Weight magnitudes
        weighted_magnitude = magnitude * gaussian_weight
        
        # Initialize descriptor
        descriptor = np.zeros(self.num_radial_bins * self.num_angular_bins * 
                            self.num_orientation_bins)
        
        # Process each spatial bin
        for r_bin in range(self.num_radial_bins):
            for a_bin in range(self.num_angular_bins):
                # Skip angular bins for center region
                if r_bin == 0 and a_bin > 0:
                    continue
                    
                # Create mask for this spatial bin
                if r_bin == 0:
                    # Center bin - no angular subdivision
                    spatial_mask = self.center_mask & self.bin_mask
                else:
                    # Ring bins - with angular subdivision
                    spatial_mask = (self.rho_bins == r_bin) & \
                                 (self.theta_bins == a_bin) & \
                                 self.bin_mask
                
                if not np.any(spatial_mask):
                    continue
                    
                # Extract orientations and magnitudes for this bin
                bin_orientations = orientation_relative[spatial_mask]
                bin_magnitudes = weighted_magnitude[spatial_mask]
                
                # Create orientation histogram
                hist, _ = np.histogram(
                    bin_orientations,
                    bins=self.num_orientation_bins,
                    range=(-np.pi, np.pi),
                    weights=bin_magnitudes
                )
                
                # Store in descriptor
                if r_bin == 0:
                    # Center bin - store in first position
                    start_idx = 0
                else:
                    # Ring bins
                    start_idx = (self.num_orientation_bins + 
                               (r_bin - 1) * self.num_angular_bins * 
                               self.num_orientation_bins + 
                               a_bin * self.num_orientation_bins)
                               
                end_idx = start_idx + self.num_orientation_bins
                descriptor[start_idx:end_idx] = hist
        
        # Normalize descriptor
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor = descriptor / norm
            
            # Threshold large values (illumination invariance)
            threshold = 0.2
            descriptor = np.minimum(descriptor, threshold)
            
            # Re-normalize
            norm = np.linalg.norm(descriptor)
            if norm > 0:
                descriptor = descriptor / norm
        
        return descriptor
        
    def extract_patch(self, image: np.ndarray, keypoint: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract patch around keypoint
        
        Args:
            image: Input image
            keypoint: Keypoint coordinates [y, x]
            
        Returns:
            Extracted patch or None if keypoint is too close to boundary
        """
        y, x = int(keypoint[0]), int(keypoint[1])
        h, w = image.shape[:2]
        
        # Check boundary conditions
        if (x < self.radius or x >= w - self.radius or 
            y < self.radius or y >= h - self.radius):
            return None
            
        # Extract patch
        patch = image[y-self.radius:y+self.radius+1, x-self.radius:x+self.radius+1]
        
        if len(patch.shape) == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            
        return patch.astype(np.float32)
        
    def describe(self, image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """
        Compute GLOH descriptors for multiple keypoints
        
        Args:
            image: Input image
            keypoints: Array of keypoints [N, 2] in format [y, x]
            
        Returns:
            Array of descriptors [N, descriptor_dim]
        """
        if len(keypoints) == 0:
            return np.empty((0, self.get_descriptor_size()))
            
        descriptors = []
        
        for keypoint in keypoints:
            patch = self.extract_patch(image, keypoint)
            
            if patch is None:
                # Keypoint too close to boundary
                descriptors.append(np.zeros(self.get_descriptor_size()))
                continue
                
            # Compute gradients
            magnitude, orientation = self.compute_gradients(patch)
            
            # Find dominant orientation
            dominant_orientation = self.compute_dominant_orientation(magnitude, orientation)
            
            # Compute descriptor
            descriptor = self.compute_descriptor(patch, dominant_orientation)
            descriptors.append(descriptor)
            
        return np.array(descriptors)
        
    def get_descriptor_size(self) -> int:
        """Get the size of GLOH descriptor"""
        # Center bin + ring bins
        center_size = self.num_orientation_bins
        ring_size = (self.num_radial_bins - 1) * self.num_angular_bins * self.num_orientation_bins
        return center_size + ring_size
