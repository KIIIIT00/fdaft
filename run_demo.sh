#!/bin/bash
# Quick FDAFT Demo Runner Script with Image Saving Support
# Usage: ./run_demo.sh [options]

set -e  # Exit on any error

echo "🌍 FDAFT Quick Demo Runner"
echo "========================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${PURPLE}[SUCCESS]${NC} $1"
}

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed or not in PATH"
    exit 1
fi

print_status "Python 3 found: $(python3 --version)"

# Check if required packages are installed
print_header "Checking Dependencies"

python3 -c "
import sys
required_packages = ['numpy', 'opencv-contrib-python', 'matplotlib', 'scipy', 'scikit-image']
missing_packages = []

for package in required_packages:
    try:
        if package == 'opencv-contrib-python':
            import cv2
            print(f'✅ OpenCV: {cv2.__version__}')
        elif package == 'scikit-image':
            import skimage
            print(f'✅ scikit-image: {skimage.__version__}')
        else:
            module = __import__(package)
            print(f'✅ {package}: {module.__version__}')
    except ImportError:
        missing_packages.append(package)
        print(f'❌ {package}: Not found')

if missing_packages:
    print(f'\\nMissing packages: {missing_packages}')
    print('Run: pip install ' + ' '.join(missing_packages))
    sys.exit(1)
else:
    print('\\n🎉 All dependencies are installed!')
"

if [ $? -ne 0 ]; then
    print_error "Dependency check failed"
    exit 1
fi

# Default settings
OUTPUT_DIR="demo_results"
ARGS=""
SAVE_IMAGES=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            echo "FDAFT Quick Demo Runner with Image Saving Support"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -h, --help              Show this help message"
            echo "  -o, --output DIR        Output directory (default: demo_results)"
            echo "  --save-images           Save visualization images (PNG files)"
            echo "  --no-display           Skip interactive display (useful for batch processing)"
            echo "  -v, --verbose           Verbose output"
            echo ""
            echo "Image Saving Options:"
            echo "  When --save-images is used, the following images will be saved:"
            echo "    - input_images_comparison.png       : Side-by-side input images"
            echo "    - matching_results_comprehensive.png: Complete matching analysis"
            echo "    - features_image1.png               : Features detected in image 1"
            echo "    - features_image2.png               : Features detected in image 2"
            echo "    - final_matches.png                 : Final matched points visualization"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Basic demo (display only)"
            echo "  $0 --save-images                     # Demo with image saving"
            echo "  $0 --save-images --no-display        # Batch mode with image saving"
            echo "  $0 -o ./my_results --save-images     # Custom output directory"
            exit 0
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            ARGS="$ARGS --output-dir $2"
            shift 2
            ;;
        --save-images)
            SAVE_IMAGES=true
            ARGS="$ARGS --save-images"
            shift
            ;;
        --no-display)
            ARGS="$ARGS --no-display"
            shift
            ;;
        -v|--verbose)
            ARGS="$ARGS --verbose"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Print demo configuration
print_header "Demo Configuration"
print_status "Output directory: $OUTPUT_DIR"
if [ "$SAVE_IMAGES" = true ]; then
    print_success "📸 Image saving: ENABLED"
    print_status "  - Input images comparison will be saved"
    print_status "  - Comprehensive matching results will be saved"
    print_status "  - Individual feature visualizations will be saved"
    print_status "  - Final matches visualization will be saved"
else
    print_status "📱 Image saving: DISABLED (use --save-images to enable)"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
print_status "Created output directory: $OUTPUT_DIR"

# Run the demo
print_header "Running FDAFT Demo"
print_status "Command: python3 demo_fdaft.py $ARGS"

# Add some spacing for better readability
echo ""

python3 demo_fdaft.py $ARGS

if [ $? -eq 0 ]; then
    echo ""
    print_header "Demo Completed Successfully!"
    print_success "✅ FDAFT demonstration finished without errors"
    print_status "📁 Check the output directory: $OUTPUT_DIR"
    
    # List the contents of output directory
    if [ -d "$OUTPUT_DIR" ]; then
        echo ""
        print_status "📋 Generated files:"
        ls -la "$OUTPUT_DIR" | while IFS= read -r line; do
            if [[ $line == *".png" ]]; then
                echo -e "${PURPLE}    🖼️  $line${NC}"
            elif [[ $line == *".txt" ]]; then
                echo -e "${BLUE}    📄  $line${NC}"
            else
                echo "    $line"
            fi
        done
    fi
    
    # Show image saving summary
    if [ "$SAVE_IMAGES" = true ]; then
        echo ""
        print_success "📸 Visualization images saved successfully!"
        print_status "  You can view the following images:"
        
        if [ -f "$OUTPUT_DIR/input_images_comparison.png" ]; then
            print_status "  ✅ Input images comparison"
        fi
        if [ -f "$OUTPUT_DIR/matching_results_comprehensive.png" ]; then
            print_status "  ✅ Comprehensive matching results"
        fi
        if [ -f "$OUTPUT_DIR/features_image1.png" ]; then
            print_status "  ✅ Features in image 1"
        fi
        if [ -f "$OUTPUT_DIR/features_image2.png" ]; then
            print_status "  ✅ Features in image 2"
        fi
        if [ -f "$OUTPUT_DIR/final_matches.png" ]; then
            print_status "  ✅ Final matches visualization"
        fi
        
        echo ""
        print_status "🔍 To view images, you can use:"
        print_status "  - Image viewer: 'open $OUTPUT_DIR/*.png' (macOS) or 'xdg-open $OUTPUT_DIR/*.png' (Linux)"
        print_status "  - Web browser: Open the PNG files in your browser"
        print_status "  - Python: Use matplotlib.image.imread() and plt.imshow()"
    fi
    
    echo ""
    print_status "🚀 Next steps:"
    print_status "  - Try with your own planetary images"
    print_status "  - Run: python scripts/extract_features.py <image_directory>"
    print_status "  - Run: python scripts/evaluate.py <dataset_directory>"
    
else
    print_error "Demo failed with exit code $?"
    print_status "Check the error messages above for troubleshooting"
    exit 1
fi

echo ""
print_success "🎉 Thank you for using FDAFT!"