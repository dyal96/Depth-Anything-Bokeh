import gradio as gr
import numpy as np
import cv2
import torch
import tempfile
import os
import subprocess
from PIL import Image
from depth_anything_v2.dpt import DepthAnythingV2

css = """
#img-display-container {
    max-height: 100vh;
}
"""

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

# Global state for caching the model
current_encoder = None
model = None

def get_encoder_from_choice(choice):
    if "Fast" in choice: return "vits"
    if "Base" in choice: return "vitb"
    return "vitl"

def load_model(encoder):
    global model, current_encoder
    if model is None or current_encoder != encoder:
        print(f"Loading Depth-Anything-V2 ({encoder})...")
        new_model = DepthAnythingV2(**model_configs[encoder])
        try:
            state_dict = torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location="cpu")
            new_model.load_state_dict(state_dict)
        except FileNotFoundError:
            raise gr.Error(f"Checkpoint for {encoder} not found! Please run the download script.")
            
        new_model = new_model.to(DEVICE).eval()
        model = new_model
        current_encoder = encoder
        print("Model loaded successfully.")
    return model

def apply_bokeh(image, depth, focal_point=0.5, max_blur_size=15, invert_depth=False):
    """
    Applies a depth-of-field (bokeh) effect to an image based on a depth map.
    """
    # Normalize depth to 0.0 - 1.0
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    
    if invert_depth:
        depth_norm = 1.0 - depth_norm
        
    blur_amount = np.abs(depth_norm - focal_point)
    levels = 10
    blurred_images = []
    max_k = int(max_blur_size) * 2 + 1
    
    for i in range(levels):
        k = int(max_k * (i / (levels - 1)))
        if k % 2 == 0:
            k += 1
            
        if k < 3:
            blurred_images.append(image.astype(np.float32))
        else:
            blurred = cv2.GaussianBlur(image, (k, k), 0)
            blurred_images.append(blurred.astype(np.float32))
            
    result = np.zeros_like(image, dtype=np.float32)
    level_map = np.clip(blur_amount * (levels - 1), 0, levels - 1)
    
    level_lower = np.floor(level_map).astype(int)
    level_upper = np.ceil(level_map).astype(int)
    blend_factor = level_map - level_lower
    blend_factor = blend_factor[..., np.newaxis]
    
    for i in range(levels):
        mask_lower = (level_lower == i)[..., np.newaxis]
        if np.any(mask_lower):
            result += mask_lower * blurred_images[i] * (1.0 - blend_factor)
            
        mask_upper = (level_upper == i)[..., np.newaxis]
        if np.any(mask_upper):
            result += mask_upper * blurred_images[i] * blend_factor
            
    return np.clip(result, 0, 255).astype(np.uint8)

def process_image(image, model_choice, focal_point, blur_size, invert_depth):
    if image is None:
        return None, None, None
        
    # Ensure model is loaded
    encoder = get_encoder_from_choice(model_choice)
    loaded_model = load_model(encoder)
        
    depth = loaded_model.infer_image(image[:, :, ::-1])
    
    depth_vis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6) * 255.0
    depth_vis = depth_vis.astype(np.uint8)
    
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    colored_depth = (cmap(depth_vis)[:, :, :3] * 255).astype(np.uint8)
    
    bokeh_image = apply_bokeh(image, depth, focal_point=focal_point, max_blur_size=blur_size, invert_depth=invert_depth)
    
    gray_depth = Image.fromarray(depth_vis)
    tmp_gray_depth = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    gray_depth.save(tmp_gray_depth.name)
    
    return bokeh_image, colored_depth, tmp_gray_depth.name

def process_video(video_path, model_choice, focal_point, blur_size, invert_depth, progress=gr.Progress()):
    if not video_path:
        return None
        
    # Ensure model is loaded
    encoder = get_encoder_from_choice(model_choice)
    loaded_model = load_model(encoder)
    
    raw_video = cv2.VideoCapture(video_path)
    frame_width = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
    total_frames = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    tmp_vid = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    tmp_vid.close()
    
    out = cv2.VideoWriter(tmp_vid.name, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))
    
    frame_idx = 0
    while raw_video.isOpened():
        ret, raw_frame = raw_video.read()
        if not ret:
            break
            
        progress(frame_idx / total_frames, desc=f"Processing Frame {frame_idx}/{total_frames}")
        
        depth = loaded_model.infer_image(raw_frame)
        bokeh_frame = apply_bokeh(raw_frame, depth, focal_point=focal_point, max_blur_size=blur_size, invert_depth=invert_depth)
        
        out.write(bokeh_frame)
        frame_idx += 1
        
    raw_video.release()
    out.release()
    
    progress(0.95, desc="Muxing audio...")
    final_vid = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    final_vid.close()
    
    subprocess.run([
        'ffmpeg', '-y', 
        '-i', tmp_vid.name, 
        '-i', video_path, 
        '-c:v', 'libx264', '-crf', '23', '-preset', 'fast',
        '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0?', 
        final_vid.name
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    try:
        os.remove(tmp_vid.name)
    except:
        pass
        
    return final_vid.name

with gr.Blocks(css=css, title="Depth-Anything-V2 Bokeh Generator") as demo:
    gr.Markdown("# 📷 Depth-of-Field (Bokeh) Generator")
    gr.Markdown("Add realistic, fake blur to your photos and videos using the **Depth-Anything-V2** model.")
    
    with gr.Row():
        model_selection = gr.Radio(
            choices=["vits (Fast)", "vitb (Base)", "vitl (High Quality)"], 
            value="vitl (High Quality)", 
            label="Model Selection (Change takes effect on next generation)"
        )

    with gr.Tabs():
        # ---------- PHOTO TAB ----------
        with gr.Tab("Photo Processing"):
            with gr.Row():
                with gr.Column():
                    photo_input = gr.Image(label="Input Image", type='numpy')
                    
                    with gr.Group():
                        gr.Markdown("### ⚙️ Adjustments")
                        p_focal_point = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="Focal Point (0.0 to 1.0)")
                        p_blur_size = gr.Slider(minimum=1, maximum=50, value=15, step=1, label="Blur Intensity")
                        p_invert_depth = gr.Checkbox(label="Invert Depth Map", value=False)
                        
                    photo_submit = gr.Button("Generate Bokeh", variant="primary")
                    
                with gr.Column():
                    photo_output = gr.Image(label="Result with Bokeh", type='numpy', format="jpeg")
                    photo_depth_vis = gr.Image(label="Generated Depth Map", type='numpy')
                    photo_depth_dl = gr.File(label="Download Grayscale Depth Map (PNG)")

            photo_submit.click(
                fn=process_image,
                inputs=[photo_input, model_selection, p_focal_point, p_blur_size, p_invert_depth],
                outputs=[photo_output, photo_depth_vis, photo_depth_dl]
            )

        # ---------- VIDEO TAB ----------
        with gr.Tab("Video Processing"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Input Video")
                    
                    with gr.Group():
                        gr.Markdown("### ⚙️ Adjustments")
                        v_focal_point = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="Focal Point (0.0 to 1.0)")
                        v_blur_size = gr.Slider(minimum=1, maximum=50, value=15, step=1, label="Blur Intensity")
                        v_invert_depth = gr.Checkbox(label="Invert Depth Map", value=False)
                        
                    video_submit = gr.Button("Generate Video Bokeh", variant="primary")
                    
                with gr.Column():
                    video_output = gr.Video(label="Result with Bokeh")

            video_submit.click(
                fn=process_video,
                inputs=[video_input, model_selection, v_focal_point, v_blur_size, v_invert_depth],
                outputs=[video_output]
            )

if __name__ == '__main__':
    # Initialize the default model before launching
    load_model("vitl")
    
    # Enable public link if running in Google Colab
    import os
    share_gradio = "COLAB_RELEASE_TAG" in os.environ
    
    demo.queue().launch(inbrowser=False, share=share_gradio)
