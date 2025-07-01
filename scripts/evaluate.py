#!/usr/bin/env python3
"""
Script to evaluate FDAFT matching performance
"""

import argparse
import os
import pickle
import numpy as np
import cv2
from pathlib import Path
import sys
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fdaft.models.fdaft import FDAFT
from fdaft.utils.visualization import FDAFTVisualizer
from fdaft.datasets.planetary_dataset import PlanetaryImageDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate FDAFT matching performance')
    parser.add_argument('dataset_dir', help='Directory containing image pairs')
    parser.add_argument('--pairs_file', help='File listing image pairs to evaluate')
    parser.add_argument('--output_dir', default='./evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--config', default='configs/fdaft/planetary.py',
                       help='Configuration file')
    parser.add_argument('--save_matches', action='store_true',
                       help='Save matching visualizations')
    parser.add_argument('--num_pairs', type=int, default=None,
                       help='Number of pairs to evaluate (for testing)')
    
    return parser.parse_args()

def evaluate_image_pair(fdaft_model, image1, image2, pair_id):
    """Evaluate matching performance on a single image pair"""
    start_time = time.time()
    
    # Perform matching
    results = fdaft_model.match_images(image1, image2)
    
    matching_time = time.time() - start_time
    
    # Compute evaluation metrics
    metrics = {
        'pair_id': pair_id,
        'matching_time': matching_time,
        'num_corner_features_1': len(results['corner_points1']),
        'num_corner_features_2': len(results['corner_points2']),
        'num_blob_features_1': len(results['blob_points1']),
        'num_blob_features_2': len(results['blob_points2']),
        'num_corner_matches': len(results['corner_matches']),
        'num_blob_matches': len(results['blob_matches']),
        'num_total_matches': len(results['all_matches_pts1']),
        'num_final_matches': results['num_final_matches'],
        'inlier_ratio': (results['num_final_matches'] / len(results['all_matches_pts1']) 
                        if len(results['all_matches_pts1']) > 0 else 0),
        'corner_match_ratio': (len(results['corner_matches']) / 
                              min(len(results['corner_points1']), len(results['corner_points2']))
                              if min(len(results['corner_points1']), len(results['corner_points2'])) > 0 else 0),
        'blob_match_ratio': (len(results['blob_matches']) / 
                            min(len(results['blob_points1']), len(results['blob_points2']))
                            if min(len(results['blob_points1']), len(results['blob_points2'])) > 0 else 0)
    }
    
    return metrics, results

def compute_summary_statistics(all_metrics):
    """Compute summary statistics across all evaluated pairs"""
    metrics_array = np.array([[
        m['matching_time'],
        m['num_final_matches'],
        m['inlier_ratio'],
        m['corner_match_ratio'],
        m['blob_match_ratio']
    ] for m in all_metrics])
    
    if len(metrics_array) == 0:
        return {}
    
    summary = {
        'num_pairs_evaluated': len(all_metrics),
        'average_matching_time': np.mean(metrics_array[:, 0]),
        'std_matching_time': np.std(metrics_array[:, 0]),
        'average_final_matches': np.mean(metrics_array[:, 1]),
        'std_final_matches': np.std(metrics_array[:, 1]),
        'average_inlier_ratio': np.mean(metrics_array[:, 2]),
        'std_inlier_ratio': np.std(metrics_array[:, 2]),
        'average_corner_match_ratio': np.mean(metrics_array[:, 3]),
        'average_blob_match_ratio': np.mean(metrics_array[:, 4]),
        'success_rate': np.mean(metrics_array[:, 1] >= 10),  # At least 10 matches
        'detailed_metrics': all_metrics
    }
    
    return summary

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration and initialize FDAFT
    if os.path.exists(args.config):
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", args.config)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config = config_module.config
    else:
        config = {"model": {}}  # Default config
    
    model_config = config.get('model', {})
    fdaft = FDAFT(**model_config)
    
    # Load dataset
    dataset = PlanetaryImageDataset(args.dataset_dir, args.pairs_file)
    
    if args.num_pairs:
        dataset.image_pairs = dataset.image_pairs[:args.num_pairs]
    
    print(f"Evaluating {len(dataset)} image pairs...")
    
    # Evaluate each pair
    all_metrics = []
    visualizer = FDAFTVisualizer()
    
    for i, sample in enumerate(dataset):
        print(f"Evaluating pair {i+1}/{len(dataset)}: {sample['image1_path']} - {sample['image2_path']}")
        
        try:
            metrics, results = evaluate_image_pair(
                fdaft, sample['image1'], sample['image2'], i
            )
            all_metrics.append(metrics)
            
            print(f"  Matches: {metrics['num_final_matches']} "
                  f"(inlier ratio: {metrics['inlier_ratio']:.2f}, "
                  f"time: {metrics['matching_time']:.2f}s)")
            
            # Save matching visualization if requested
            if args.save_matches:
                vis_file = output_dir / f"matches_pair_{i:03d}.png"
                visualizer.plot_matching_results(results, sample['image1'], sample['image2'])
                plt.savefig(vis_file, dpi=150, bbox_inches='tight')
                plt.close()
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Compute and save summary statistics
    summary = compute_summary_statistics(all_metrics)
    
    # Save results
    results_file = output_dir / 'evaluation_results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump({
            'summary': summary,
            'config': config,
            'args': vars(args)
        }, f)
    
    # Print summary
    print(f"\nEvaluation Summary# FDAFT Project Structure and Implementation")
    print(f"\nEvaluation Summary:")
    print(f"=" * 50)
    print(f"Pairs evaluated: {summary['num_pairs_evaluated']}")
    print(f"Average matching time: {summary['average_matching_time']:.2f} ± {summary['std_matching_time']:.2f} seconds")
    print(f"Average final matches: {summary['average_final_matches']:.1f} ± {summary['std_final_matches']:.1f}")
    print(f"Average inlier ratio: {summary['average_inlier_ratio']:.2f} ± {summary['std_inlier_ratio']:.2f}")
    print(f"Success rate (≥10 matches): {summary['success_rate']:.2f}")
    print(f"Average corner match ratio: {summary['average_corner_match_ratio']:.2f}")
    print(f"Average blob match ratio: {summary['average_blob_match_ratio']:.2f}")
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()