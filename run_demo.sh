#!/bin/bash
# Quick FDAFT Demo Runner Script
# Usage: ./run_demo.sh [options]

set -e  # Exit on any error

echo "🌍 FDAFT Quick Demo Runner"
echo "========================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Default demo mode
DEMO_MODE="synthetic"
OUTPUT_DIR="demo_results"
ARGS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            echo "FDAFT Quick Demo Runner"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -h, --help              Show this help message"
            echo "  -s, --synthetic         Use synthetic images (default)"
            echo "  -i, --interactive       Interactive file selection"
            echo "  -d, --directory DIR     Process all images in directory"
            echo "  -1, --image1 FILE       First image file"
            echo "  -2, --image2 FILE       Second image file"
            echo "  -o, --output DIR        Output directory (default: demo_results)"
            echo "  -b, --benchmark         Run performance benchmark"
            echo "  -v, --verbose           Verbose output"
            echo "  --no-viz               Skip visualization"
            echo "  --save                 Save results to files"
            echo "  --enhance              Apply contrast enhancement"
            echo "  --resize WxH           Resize images (e.g., 512x512)"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Synthetic demo"
            echo "  $0 -i                                 # Interactive mode"
            echo "  $0 -d ./images                       # Process directory"
            echo "  $0 -1 mars1.jpg -2 mars2.jpg         # Specific images"
            echo "  $0 -b --save                         # Benchmark with save"
            exit 0
            ;;
        -s|--synthetic)
            DEMO_MODE="synthetic"
            ARGS="$ARGS --synthetic"
            shift
            ;;
        -i|--interactive)
            DEMO_MODE="interactive"
            ARGS="$ARGS --interactive"
            shift
            ;;
        -d|--directory)
            DEMO_MODE="directory"
            ARGS="$ARGS --directory $2"
            shift 2
            ;;
        -1|--image1)
            IMAGE1="$2"
            ARGS="$ARGS --image1 $2"
            shift 2
            ;;
        -2|--image2)
            IMAGE2="$2"
            ARGS="$ARGS --image2 $2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            ARGS="$ARGS --output-dir $2"
            shift 2
            ;;
        -b|--benchmark)
            ARGS="$ARGS --benchmark"
            shift
            ;;
        -v|--verbose)
            ARGS="$ARGS --verbose"
            shift
            ;;
        --no-viz)
            ARGS="$ARGS --no-visualization"
            shift
            ;;
        --save)
            ARGS="$ARGS --save-results"
            shift
            ;;
        --enhance)
            ARGS="$ARGS --enhance-contrast"
            shift
            ;;
        --resize)
            ARGS="$ARGS --resize $2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Validate specific image inputs
if [ ! -z "$IMAGE1" ] && [ -z "$IMAGE2" ]; then
    print_error "When using --image1, --image2 is also required"
    exit 1
fi

if [ ! -z "$IMAGE1" ]; then
    if [ ! -f "$IMAGE1" ]; then
        print_error "Image file not found: $IMAGE1"
        exit 1
    fi
    DEMO_MODE="specific"
fi

if [ ! -z "$IMAGE2" ]; then
    if [ ! -f "$IMAGE2" ]; then
        print_error "Image file not found: $IMAGE2"
        exit 1
    fi
fi

# Print demo configuration
print_header "Demo Configuration"
print_status "Mode: $DEMO_MODE"
print_status "Output directory: $OUTPUT_DIR"
if [ ! -z "$IMAGE1" ]; then
    print_status "Image 1: $IMAGE1"
    print_status "Image 2: $IMAGE2"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the demo
print_header "Running FDAFT Demo"
print_status "Command: python3 fdaft_demo.py $ARGS"

python3 fdaft_demo.py $ARGS

if [ $? -eq 0 ]; then
    print_header "Demo Completed Successfully!"
    print_status "Check the output directory: $OUTPUT_DIR"
else
    print_error "Demo failed with exit code $?"
    exit 1
fi