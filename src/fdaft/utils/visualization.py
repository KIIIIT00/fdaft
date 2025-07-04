import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import Dict, Optional

class FDAFTVisualizer:
    """Visualization utilities for FDAFT results with saving capability"""
    
    @staticmethod
    def plot_features(image: np.ndarray, corner_points: np.ndarray, blob_points: np.ndarray,
                     title: str = "FDAFT Features", figsize: tuple = (12, 8), save_path: Optional[str] = None):
        """
        Plot extracted features on image
        
        Args:
            image: Input image
            corner_points: Corner point coordinates [N, 2]
            blob_points: Blob point coordinates [M, 2]
            title: Plot title
            figsize: Figure size
            save_path: Path to save the figure (optional)
        """
        plt.figure(figsize=figsize)
        
        # Display image
        if len(image.shape) == 3:
            plt.imshow(image)
        else:
            plt.imshow(image, cmap='gray')
        
        # Plot corner points (red circles)
        if len(corner_points) > 0:
            plt.scatter(corner_points[:, 1], corner_points[:, 0], 
                       c='red', s=30, marker='o', alpha=0.7, 
                       label=f'Corner Points ({len(corner_points)})')
        
        # Plot blob points (blue triangles)
        if len(blob_points) > 0:
            plt.scatter(blob_points[:, 1], blob_points[:, 0], 
                       c='blue', s=30, marker='^', alpha=0.7, 
                       label=f'Blob Points ({len(blob_points)})')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc='upper right')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Features plot saved to: {save_path}")
        else:
            plt.show()
    
    @staticmethod
    def plot_matches(image1: np.ndarray, image2: np.ndarray, 
                    points1: np.ndarray, points2: np.ndarray,
                    title: str = "FDAFT Matches", figsize: tuple = (16, 8), save_path: Optional[str] = None):
        """
        Plot matches between two images
        
        Args:
            image1, image2: Input images
            points1, points2: Matched points
            title: Plot title
            figsize: Figure size
            save_path: Path to save the figure (optional)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Display images
        if len(image1.shape) == 3:
            ax1.imshow(image1)
            ax2.imshow(image2)
        else:
            ax1.imshow(image1, cmap='gray')
            ax2.imshow(image2, cmap='gray')
        
        # Plot points
        if len(points1) > 0:
            ax1.scatter(points1[:, 1], points1[:, 0], c='red', s=20, alpha=0.7)
            ax2.scatter(points2[:, 1], points2[:, 0], c='red', s=20, alpha=0.7)
        
        ax1.set_title(f'Image 1 ({len(points1)} points)', fontsize=12, fontweight='bold')
        ax2.set_title(f'Image 2 ({len(points2)} points)', fontsize=12, fontweight='bold')
        ax1.axis('off')
        ax2.axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Matches plot saved to: {save_path}")
        else:
            plt.show()
    
    @staticmethod
    def plot_matching_results(results: Dict, image1: np.ndarray, image2: np.ndarray, save_path: Optional[str] = None):
        """
        Plot comprehensive matching results
        
        Args:
            results: Results dictionary from FDAFT.match_images()
            image1, image2: Input images
            save_path: Path to save the figure (optional)
        """
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Features in image 1
        plt.subplot(2, 3, 1)
        if len(image1.shape) == 3:
            plt.imshow(image1)
        else:
            plt.imshow(image1, cmap='gray')
        
        if len(results['corner_points1']) > 0:
            plt.scatter(results['corner_points1'][:, 1], results['corner_points1'][:, 0], 
                       c='red', s=20, alpha=0.7, label=f"Corners ({len(results['corner_points1'])})")
        if len(results['blob_points1']) > 0:
            plt.scatter(results['blob_points1'][:, 1], results['blob_points1'][:, 0], 
                       c='blue', s=20, alpha=0.7, label=f"Blobs ({len(results['blob_points1'])})")
        plt.title('Image 1 - Features', fontsize=12, fontweight='bold')
        plt.legend(loc='upper right')
        plt.axis('off')
        
        # 2. Features in image 2
        plt.subplot(2, 3, 2)
        if len(image2.shape) == 3:
            plt.imshow(image2)
        else:
            plt.imshow(image2, cmap='gray')
        
        if len(results['corner_points2']) > 0:
            plt.scatter(results['corner_points2'][:, 1], results['corner_points2'][:, 0], 
                       c='red', s=20, alpha=0.7, label=f"Corners ({len(results['corner_points2'])})")
        if len(results['blob_points2']) > 0:
            plt.scatter(results['blob_points2'][:, 1], results['blob_points2'][:, 0], 
                       c='blue', s=20, alpha=0.7, label=f"Blobs ({len(results['blob_points2'])})")
        plt.title('Image 2 - Features', fontsize=12, fontweight='bold')
        plt.legend(loc='upper right')
        plt.axis('off')
        
        # 3. All matches
        plt.subplot(2, 3, 3)
        FDAFTVisualizer._plot_match_lines(image1, image2, 
                                         results['all_matches_pts1'], 
                                         results['all_matches_pts2'],
                                         'All Matches')
        
        # 4. Filtered matches
        plt.subplot(2, 3, 4)
        FDAFTVisualizer._plot_match_lines(image1, image2, 
                                         results['filtered_matches_pts1'], 
                                         results['filtered_matches_pts2'],
                                         'Filtered Matches (RANSAC)')
        
        # 5. Match statistics
        plt.subplot(2, 3, 5)
        labels = ['Corner\nMatches', 'Blob\nMatches', 'Total\nMatches', 'Final\nMatches']
        values = [len(results['corner_matches']), len(results['blob_matches']), 
                 len(results['all_matches_pts1']), results['num_final_matches']]
        colors = ['red', 'blue', 'green', 'orange']
        
        bars = plt.bar(labels, values, color=colors, alpha=0.7)
        plt.title('Matching Statistics', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Matches')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
                    str(value), ha='center', va='bottom', fontweight='bold')
        
        # Set y-axis limit with some padding
        plt.ylim(0, max(values) * 1.15)
        
        # 6. Match quality metrics
        plt.subplot(2, 3, 6)
        if len(results['all_matches_pts1']) > 0:
            inlier_ratio = results['num_final_matches'] / len(results['all_matches_pts1'])
        else:
            inlier_ratio = 0
        
        metrics = ['Inlier Ratio', 'Corner/Total', 'Blob/Total']
        if len(results['all_matches_pts1']) > 0:
            corner_ratio = len(results['corner_matches']) / len(results['all_matches_pts1'])
            blob_ratio = len(results['blob_matches']) / len(results['all_matches_pts1'])
        else:
            corner_ratio = blob_ratio = 0
            
        values = [inlier_ratio, corner_ratio, blob_ratio]
        
        bars = plt.bar(metrics, values, color=['orange', 'red', 'blue'], alpha=0.7)
        plt.title('Match Quality Metrics', fontsize=12, fontweight='bold')
        plt.ylabel('Ratio')
        plt.ylim(0, 1)
        
        # Add value labels
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Add overall title
        plt.suptitle('FDAFT Matching Results - Comprehensive Analysis', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Comprehensive matching results saved to: {save_path}")
        else:
            plt.show()
    
    @staticmethod
    def _plot_match_lines(image1: np.ndarray, image2: np.ndarray, 
                         points1: np.ndarray, points2: np.ndarray, title: str):
        """Helper function to plot matches with connecting lines"""
        # Create side-by-side image
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        h = max(h1, h2)
        w = w1 + w2
        
        if len(image1.shape) == 3:
            combined = np.zeros((h, w, 3), dtype=image1.dtype)
            combined[:h1, :w1] = image1
            combined[:h2, w1:w1+w2] = image2
        else:
            combined = np.zeros((h, w), dtype=image1.dtype)
            combined[:h1, :w1] = image1
            combined[:h2, w1:w1+w2] = image2
        
        plt.imshow(combined, cmap='gray' if len(combined.shape) == 2 else None)
        
        # Plot matches and connecting lines
        if len(points1) > 0:
            # Points in image 1
            plt.scatter(points1[:, 1], points1[:, 0], c='red', s=15, alpha=0.8, edgecolors='white', linewidth=0.5)
            # Points in image 2 (shifted by w1)
            plt.scatter(points2[:, 1] + w1, points2[:, 0], c='red', s=15, alpha=0.8, edgecolors='white', linewidth=0.5)
            
            # Connecting lines
            for i in range(len(points1)):
                plt.plot([points1[i, 1], points2[i, 1] + w1], 
                        [points1[i, 0], points2[i, 0]], 
                        'g-', alpha=0.6, linewidth=1)
        
        plt.title(f'{title} ({len(points1)} matches)', fontsize=12, fontweight='bold')
        plt.axis('off')
    
    @staticmethod
    def plot_feature_distribution(results: Dict, save_path: Optional[str] = None):
        """
        Plot feature distribution analysis
        
        Args:
            results: Results dictionary from FDAFT.match_images()
            save_path: Path to save the figure (optional)
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Feature counts per image
        feature_types = ['Corners', 'Blobs']
        image1_counts = [len(results['corner_points1']), len(results['blob_points1'])]
        image2_counts = [len(results['corner_points2']), len(results['blob_points2'])]
        
        x = np.arange(len(feature_types))
        width = 0.35
        
        ax1.bar(x - width/2, image1_counts, width, label='Image 1', color='lightblue', alpha=0.7)
        ax1.bar(x + width/2, image2_counts, width, label='Image 2', color='lightcoral', alpha=0.7)
        ax1.set_title('Feature Counts by Type', fontweight='bold')
        ax1.set_xlabel('Feature Type')
        ax1.set_ylabel('Count')
        ax1.set_xticks(x)
        ax1.set_xticklabels(feature_types)
        ax1.legend()
        
        # Add value labels on bars
        for i, (v1, v2) in enumerate(zip(image1_counts, image2_counts)):
            ax1.text(i - width/2, v1 + max(image1_counts + image2_counts)*0.01, str(v1), 
                    ha='center', va='bottom', fontweight='bold')
            ax1.text(i + width/2, v2 + max(image1_counts + image2_counts)*0.01, str(v2), 
                    ha='center', va='bottom', fontweight='bold')
        
        # Match counts
        match_types = ['Corner\nMatches', 'Blob\nMatches', 'Total\nMatches', 'Final\nMatches']
        match_counts = [len(results['corner_matches']), len(results['blob_matches']),
                       len(results['all_matches_pts1']), results['num_final_matches']]
        colors = ['red', 'blue', 'green', 'orange']
        
        bars = ax2.bar(match_types, match_counts, color=colors, alpha=0.7)
        ax2.set_title('Matching Performance', fontweight='bold')
        ax2.set_ylabel('Number of Matches')
        
        for bar, count in zip(bars, match_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(match_counts)*0.01,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # Match ratios
        if len(results['all_matches_pts1']) > 0:
            inlier_ratio = results['num_final_matches'] / len(results['all_matches_pts1'])
            corner_ratio = len(results['corner_matches']) / len(results['all_matches_pts1'])
            blob_ratio = len(results['blob_matches']) / len(results['all_matches_pts1'])
        else:
            inlier_ratio = corner_ratio = blob_ratio = 0
            
        ratio_types = ['Inlier\nRatio', 'Corner\nContribution', 'Blob\nContribution']
        ratio_values = [inlier_ratio, corner_ratio, blob_ratio]
        
        bars = ax3.bar(ratio_types, ratio_values, color=['orange', 'red', 'blue'], alpha=0.7)
        ax3.set_title('Match Quality Ratios', fontweight='bold')
        ax3.set_ylabel('Ratio')
        ax3.set_ylim(0, 1)
        
        for bar, ratio in zip(bars, ratio_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{ratio:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Success metrics pie chart
        if results['num_final_matches'] > 0:
            success_labels = ['Successful\nMatches', 'Filtered Out\nMatches']
            success_sizes = [results['num_final_matches'], 
                           len(results['all_matches_pts1']) - results['num_final_matches']]
            success_colors = ['lightgreen', 'lightcoral']
            
            ax4.pie(success_sizes, labels=success_labels, colors=success_colors, autopct='%1.1f%%',
                   startangle=90, textprops={'fontweight': 'bold'})
            ax4.set_title('Match Success Rate', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'No matches found', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=14, fontweight='bold')
            ax4.set_title('Match Success Rate', fontweight='bold')
        
        plt.suptitle('FDAFT Feature Distribution Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Feature distribution plot saved to: {save_path}")
        else:
            plt.show()