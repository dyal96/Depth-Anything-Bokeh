# Depth-Anything-Bokeh

A simple, optimized, and standalone Gradio application for applying highly realistic depth-of-field (bokeh) effects to your photos and videos, powered by [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2).

This standalone application was customized to allow quick, interactive depth-based blurring, supporting both images and `.mp4` video files.

## Features
- **Interactive UI**: Upload images/videos and adjust the focal point and blur intensity on the fly.
- **Model Selection**: Switch between Fast (vits), Base (vitb), and High Quality (vitl) Depth-Anything-V2 models directly within the UI.
- **Video Support**: Automatically extracts video frames, applies the depth blur, and uses `ffmpeg` to stitch everything back together with the original audio.
- **Depth Map Export**: Easily download the raw grayscale depth map of your photos.

## Requirements
- Python 3.10+
- `ffmpeg` installed on your system (Required for preserving audio in videos)

## Quick Setup (Windows)
1. Double-click `install.bat`. This will automatically:
   - Create a Python virtual environment.
   - Install all required dependencies.
   - Download the model checkpoints (Fast, Base, and High Quality).
2. Double-click `start_app.bat` to launch the Gradio UI!

## Quick Setup (Linux / Mac)
1. Make the scripts executable:
   ```bash
   chmod +x install.sh start_app.sh
   ```
2. Run the install script:
   ```bash
   ./install.sh
   ```
3. Run the application:
   ```bash
   ./start_app.sh
   ```

## Usage Notes
- The models range in size from ~99 MB to ~1.3 GB. The `install.bat` / `install.sh` will attempt to download all three so you can switch between them in the UI freely.
- Processing videos frame-by-frame takes time depending on your GPU. The Fast model (`vits`) is highly recommended for videos if you want quicker results.

## Acknowledgements
Powered by the incredible [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) project.
