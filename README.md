# FDAFT Demo Setup and Usage Guide

## Quick Start

### 1. Environment Setup

```bash
# Clone or navigate to FDAFT project directory
cd FDAFT

# Install dependencies
pip install -r requirements.txt

# Install FDAFT in development mode
pip install -e .

# Make demo scripts executable (Unix/Mac)
chmod +x run_demo.sh
chmod +x fdaft_demo.py
chmod +x batch_demo.py
```

### 2. Run Basic Demo

```bash
# Method 1: Using the quick runner script
./run_demo.sh

# Method 2: Direct Python execution
python fdaft_demo.py
```

## Demo Modes

### 🎨 Synthetic Images (Default)
```bash
# Basic synthetic demo
python fdaft_demo.py --synthetic

# Synthetic demo with results saving
python fdaft_demo.py --synthetic --save-results
```

### 📁 Interactive File Selection
```bash
# Interactive mode (opens file dialog)
python fdaft_demo.py --interactive

# Using the runner script
./run_demo.sh -i
```

### 🖼️ Specific Image Pair
```bash
# Process specific images
python fdaft_demo.py --image1 mars_image1.jpg --image2 mars_image2.jpg

# With preprocessing
python fdaft_demo.py --image1 img1.png --image2 img2.png --enhance-contrast --resize 512 512
```

### 📂 Directory Processing
```bash
# Process all images in directory
python fdaft_demo.py --directory ./planetary_images

# Directory with custom settings
python fdaft_demo.py --directory ./images --max-keypoints 1500 --save-results
```

## Advanced Usage

### 🔧 Parameter Tuning
```bash
# Adjust FDAFT parameters
python fdaft_demo.py \
    --max-keypoints 2000 \
    --descriptor-radius 48 \
    --num-layers 4 \
    --save-results

# High-quality processing
python fdaft_demo.py \
    --image1 high_res_mars1.tif \
    --image2 high_res_mars2.tif \
    --max-keypoints 3000 \
    --descriptor-radius 64 \
    --enhance-contrast
```

### 📊 Performance Benchmarking
```bash
# Run performance benchmark
python fdaft_demo.py --benchmark --save-results

# Benchmark with specific images
python fdaft_demo.py --image1 test1.jpg --image2 test2.jpg --benchmark
```

### 🎯 Batch Processing
```bash
# Process multiple image pairs
python batch_demo.py --input-dir ./dataset --output-dir ./results

# Process from pairs file
python batch_demo.py --pairs-file image_pairs.txt --output-dir ./results

# Generate synthetic dataset and process
python batch_demo.py --generate-synthetic 50 --output-dir ./synthetic_results

# Batch with comprehensive analysis
python batch_demo.py \
    --input-dir ./large_dataset \
    --output-dir ./analysis_results \
    --save-visualizations \
    --save-features \
    --generate-report \
    --benchmark
```

## Command Line Options

### Basic Options
- `--image1 FILE`: First image file
- `--image2 FILE`: Second image file  
- `--directory DIR`: Process all images in directory
- `--interactive`: Interactive file selection
- `--synthetic`: Use synthetic images (default)

### Processing Options
- `--max-keypoints N`: Maximum keypoints to extract (default: 800)
- `--descriptor-radius N`: GLOH descriptor radius (default: 32)
- `--num-layers N`: Scale space layers (default: 3)
- `--resize W H`: Resize images to W×H pixels
- `--enhance-contrast`: Apply contrast enhancement

### Output Options
- `--output-dir DIR`: Output directory (default: demo_results)
- `--save-results`: Save results to files
- `--no-visualization`: Skip visualization (for batch processing)

### Performance Options
- `--benchmark`: Run performance benchmark
- `--verbose`: Verbose output

## File Formats

### Supported Image Formats
- **JPEG**: `.jpg`, `.jpeg`
- **PNG**: `.png`
- **TIFF**: `.tif`, `.tiff`
- **Bitmap**: `.bmp`

### Image Pairs File Format
Create a text file with image pairs (one pair per line):
```
mars_ctx_001.jpg mars_ctx_002.jpg
lunar_nav_01.png lunar_nav_02.png
# Comments start with #
crater_sequence_1.tif crater_sequence_2.tif
```

## Output Structure

When using `--save-results`, the output directory will contain:

```
demo_results/
├── pair_001/
│   ├── input_image1.png
│   ├── input_image2.png
│   ├── corner_points1.txt
│   ├── corner_points2.txt
│   ├── blob_points1.txt
│   ├── blob_points2.txt
│   ├── corner_matches.txt
│   ├── blob_matches.txt
│   ├── final_matches_pts1.txt
│   ├── final_matches_pts2.txt
│   └── summary_report.txt
├── pair_002/
│   └── ...
└── benchmark_results.txt (if --benchmark used)
```

## Examples by Use Case

### 🌙 Lunar Image Analysis
```bash
# Lunar images often have low contrast
python fdaft_demo.py \
    --image1 lunar_surface_1.tif \
    --image2 lunar_surface_2.tif \
    --enhance-contrast \
    --max-keypoints 1500 \
    --descriptor-radius 40
```

### 🔴 Mars Terrain Matching
```bash
# Mars images with good texture
python fdaft_demo.py \
    --image1 mars_hirise_1.jpg \
    --image2 mars_hirise_2.jpg \
    --max-keypoints 2000 \
    --num-layers 4
```

### 🛰️ Satellite Image Processing
```bash
# High-resolution satellite imagery
python fdaft_demo.py \
    --directory ./satellite_images \
    --resize 1024 1024 \
    --max-keypoints 3000 \
    --save-results
```

### 🧪 Research and Evaluation
```bash
# Comprehensive evaluation
python batch_demo.py \
    --input-dir ./research_dataset \
    --output-dir ./evaluation_results \
    --save-visualizations \
    --save-features \
    --generate-report \
    --benchmark \
    --max-pairs 100
```

## Troubleshooting

### Common Issues

1. **Import Error**:
   ```bash
   # Reinstall FDAFT
   pip uninstall fdaft
   pip install -e .
   ```

2. **OpenCV Issues**:
   ```bash
   # Install OpenCV with contrib modules
   pip uninstall opencv-python
   pip install opencv-contrib-python>=4.5.0
   ```

3. **Memory Issues**:
   ```bash
   # Reduce image size and keypoints
   python fdaft_demo.py --resize 512 512 --max-keypoints 500
   ```

4. **Display Issues**:
   ```bash
   # Use non-interactive mode
   python fdaft_demo.py --no-visualization --save-results
   ```

### Performance Optimization

- **For Speed**: Reduce `--max-keypoints` and `--descriptor-radius`
- **For Accuracy**: Increase `--max-keypoints` and `--num-layers`
- **For Memory**: Use `--resize` to limit image dimensions

### Getting Help

```bash
# Show help for main demo
python fdaft_demo.py --help

# Show help for batch processing
python batch_demo.py --help

# Show help for runner script
./run_demo.sh --help
```

## Integration with Other Tools

### Using Results in MATLAB
```matlab
% Load feature points
corner_pts = readmatrix('corner_points1.txt');
blob_pts = readmatrix('blob_points1.txt');

% Load matches
matches = readmatrix('final_matches_pts1.txt');
```

### Using Results in Python
```python
import numpy as np

# Load results
corner_points = np.loadtxt('corner_points1.txt')
matches_pts1 = np.loadtxt('final_matches_pts1.txt')
matches_pts2 = np.loadtxt('final_matches_pts2.txt')

# Visualize with matplotlib
import matplotlib.pyplot as plt
plt.scatter(corner_points[:, 1], corner_points[:, 0])
plt.show()
```

## Performance Expectations

### Typical Performance (512×512 images)
- **Processing Time**: 2-5 seconds per image pair
- **Feature Count**: 500-1500 features per image
- **Match Count**: 50-500 final matches
- **Success Rate**: 80-95% on suitable planetary images

### Hardware Recommendations
- **Minimum**: 4GB RAM, dual-core CPU
- **Recommended**: 8GB+ RAM, quad-core+ CPU
- **For Batch Processing**: 16GB+ RAM, multi-core CPU

---

**Ready to explore planetary image matching with FDAFT!** 🚀

For more information, see the main [README.md](README.md) and [documentation](docs/).