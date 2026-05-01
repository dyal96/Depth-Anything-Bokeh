@echo off
echo Setting up Depth-Anything-Bokeh...

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing dependencies...
pip install -r requirements.txt
pip install xformers
pip install git+https://github.com/ChaoningZhang/MobileSAM.git

if not exist "checkpoints" (
    mkdir checkpoints
)

echo.
echo =======================================================
echo Downloading Checkpoints...
echo This might take a while depending on your internet speed.
echo =======================================================

if not exist "checkpoints\depth_anything_v2_vits.pth" (
    echo Downloading Fast Model (vits) ~99MB...
    curl -L -o checkpoints\depth_anything_v2_vits.pth https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
) else (
    echo vits checkpoint already exists.
)

if not exist "checkpoints\depth_anything_v2_vitb.pth" (
    echo Downloading Base Model (vitb) ~390MB...
    curl -L -o checkpoints\depth_anything_v2_vitb.pth https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth
) else (
    echo vitb checkpoint already exists.
)

if not exist "checkpoints\depth_anything_v2_vitl.pth" (
    echo Downloading High Quality Model (vitl) ~1.3GB...
    curl -L -o checkpoints\depth_anything_v2_vitl.pth https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
) else (
    echo vitl checkpoint already exists.
)

if not exist "checkpoints\mobile_sam.pt" (
    echo Downloading MobileSAM weights ~40MB...
    curl -L -o checkpoints\mobile_sam.pt https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
) else (
    echo mobile_sam weights already exist.
)

if not exist "checkpoints\yolov8n-seg.pt" (
    echo Downloading YOLOv8-seg weights ~7MB...
    curl -L -o checkpoints\yolov8n-seg.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.pt
) else (
    echo yolov8n-seg weights already exist.
)

echo.
echo =======================================================
echo Installation Complete! 
echo Run start_app.bat to launch the application.
echo =======================================================
pause
