# Configuration for planetary image matching

config = {
    "model": {
        "num_layers": 3,
        "sigma_0": 1.0,
        "descriptor_radius": 48,
        "max_keypoints": 1000,
        "nms_radius": 5,
    },
    
    "scale_space": {
        "ecm_patch_size": 16,
        "steerable_filter_sigma": 2.0,
        "phase_congruency": {
            "orientations": [0, 30, 60, 90, 120, 150],  # degrees
            "scales": [1, 2, 4],
            "noise_threshold": 0.1,
        }
    },
    
    "feature_detector": {
        "fast_threshold": 0.05,
        "blob_threshold": 0.05,
        "corner_k": 0.04,  # Harris parameter
    },
    
    "descriptor": {
        "gloh_radial_bins": 3,
        "gloh_angular_bins": 8,
        "gloh_orientation_bins": 16,
        "gradient_threshold": 0.1,
        "lambda_ori": 1.5,
        "lambda_desc": 6.0,
    },
    
    "matching": {
        "ratio_threshold": 0.8,  # Lowe's ratio test
        "ransac_threshold": 3.0,  # pixels
        "ransac_max_iters": 1000,
    },
    
    "planetary_specific": {
        "crater_detection": True,
        "circular_structure_radius_range": [10, 100],
        "texture_mask_threshold": 0.02,
        "contrast_enhancement": True,
    }
}