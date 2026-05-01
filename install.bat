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

echo.
echo =======================================================
echo Installation Complete! 
echo Run start_app.bat to launch the application.
echo =======================================================
pause
