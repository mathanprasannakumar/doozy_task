#!/bin/bash
set -e  # exit immediately if a command fails

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if uv is installed
if command_exists uv; then
    echo "uv is already installed."
else
    echo "uv not found. Installing..."
    pip install uv
fi

# Clone the repo if not already present
if [ ! -d "happypose" ]; then
    git clone --branch dev --recurse-submodules https://github.com/agimus-project/happypose.git
else
    echo "happypose folder already exists. Skipping clone."
fi

# Move into the project
cd happypose || exit 1

# Sync dependencies (choose CPU or CUDA version)
uv sync --extra pypi --extra cpu   # change cpu -> cu124 if you want CUDA

# Activate virtual environment
source .venv/bin/activate

# Create load_data directory
mkdir -p "$(pwd)/load_data"

# Set environment variable
export HAPPYPOSE_DATA_DIR="$(pwd)/load_data"
echo "HAPPYPOSE_DATA_DIR set to: $HAPPYPOSE_DATA_DIR"

echo "HappyPose environment is ready!"
echo "You are now inside the virtual environment."

# Download MegaPose models
python -m happypose.toolbox.utils.download --megapose_models
echo "MegaPose models downloaded successfully into $HAPPYPOSE_DATA_DIR"

# Download example dataset (barbecue-sauce)
python -m happypose.toolbox.utils.download --examples barbecue-sauce
echo "Example dataset 'barbecue-sauce' downloaded into $HAPPYPOSE_DATA_DIR"

# Create hex_bolt_30 dataset structure
EXAMPLE_DIR="$HAPPYPOSE_DATA_DIR/examples/hex_bolt_30"
mkdir -p "$EXAMPLE_DIR/data" \
         "$EXAMPLE_DIR/meshes" \
         "$EXAMPLE_DIR/model" \
         "$EXAMPLE_DIR/outputs" \
         "$EXAMPLE_DIR/pose_output" \
         "$EXAMPLE_DIR/visualization"
cd ..

# Run STL → PLY conversion
if [ -f "stl2plyconv.py" ]; then
    echo " Converting STL to PLY..."
    python stl2plyconv.py
    echo "STL conversion finished."
else
    echo "stl2plyconv.py not found in parent directory!"
fi

# Copy PLY file into meshes folder
if [ -f "hex_bolt_30.ply" ]; then
    cp hex_bolt_30.ply "$EXAMPLE_DIR/meshes/"
    echo "hex_bolt_30.ply copied to $EXAMPLE_DIR/meshes/"
else
    echo "hex_bolt_30.ply not found!"
fi

if [ -f "../blender/camera_data.json" ]; then
    cp ../blender/camera_data.json "$EXAMPLE_DIR/"
    echo "camera_data.json copied to $EXAMPLE_DIR"
else
    echo "../blender/camera_data.json not found!"
fi

if [ -f "pose_inference.py" ]; then
    cp pose_inference.py "$EXAMPLE_DIR/"
    echo "pose_inference.py copied to $EXAMPLE_DIR"
else
    echo "pose_inference.py not found!"
fi

cd $EXAMPLE_DIR

