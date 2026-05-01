#!/bin/bash
echo "Setting up Depth-Anything-Bokeh..."

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt
pip install xformers
pip install git+https://github.com/ChaoningZhang/MobileSAM.git

if [ ! -d "checkpoints" ]; then
    mkdir checkpoints
fi

echo ""
echo "======================================================="
echo "Downloading Checkpoints..."
echo "This might take a while depending on your internet speed."
echo "======================================================="

if [ ! -f "checkpoints/depth_anything_v2_vits.pth" ]; then
    echo "Downloading Fast Model (vits) ~99MB..."
    curl -L -o checkpoints/depth_anything_v2_vits.pth https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
else
    echo "vits checkpoint already exists."
fi

if [ ! -f "checkpoints/depth_anything_v2_vitb.pth" ]; then
    echo "Downloading Base Model (vitb) ~390MB..."
    curl -L -o checkpoints/depth_anything_v2_vitb.pth https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth
else
    echo "vitb checkpoint already exists."
fi

if [ ! -f "checkpoints/depth_anything_v2_vitl.pth" ]; then
    echo "Downloading High Quality Model (vitl) ~1.3GB..."
    curl -L -o checkpoints/depth_anything_v2_vitl.pth https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
else
    echo "vitl checkpoint already exists."
fi

if [ ! -f "checkpoints/mobile_sam.pt" ]; then
    echo "Downloading MobileSAM weights ~40MB..."
    curl -L -o checkpoints/mobile_sam.pt https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
else
    echo "mobile_sam.pt already exists."
fi

if [ ! -f "checkpoints/yolov8n-seg.pt" ]; then
    echo "Downloading YOLOv8-seg weights ~7MB..."
    curl -L -o checkpoints/yolov8n-seg.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.pt
else
    echo "yolov8n-seg.pt already exists."
fi

echo ""
echo "======================================================="
echo "Installation Complete!"
echo "Run ./start_app.sh to launch the application."
echo "======================================================="
