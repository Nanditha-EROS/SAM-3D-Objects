"""
SAM 3D Objects - Gradio App
Upload image → Generate masks → Select object → Get 3D model
"""

import gradio as gr
import numpy as np
import cv2
import os
import sys
import uuid

sys.path.append('notebook')
from inference import Inference

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Global models
inference_model = None
sam_model = None


def load_sam_model():
    """Load SAM2 model"""
    global sam_model
    if sam_model is None:
        print("Loading SAM2 model...")
        sam2_checkpoint = "sam2.1_hiera_small.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        sam2 = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
        sam_model = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=32,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.92,
            min_mask_region_area=100,
        )
        print("SAM2 model loaded!")
    return sam_model


def load_3d_model():
    """Load 3D inference model"""
    global inference_model
    if inference_model is None:
        print("Loading 3D model...")
        config_path = 'checkpoints/hf/pipeline.yaml'
        inference_model = Inference(config_path, compile=False)
        print("3D model loaded!")
    return inference_model


def generate_masks(image):
    """Generate masks from uploaded image and return gallery"""
    if image is None:
        return [], None, "❌ Please upload an image"
    
    # Convert to RGB
    if image.ndim == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 4:
        image_rgb = image[:, :, :3]
    else:
        image_rgb = image
    
    # Save image temporarily
    temp_dir = "temp_masks"
    os.makedirs(temp_dir, exist_ok=True)
    cv2.imwrite(f"{temp_dir}/image.png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    
    # Generate masks
    sam = load_sam_model()
    print("Generating masks...")
    masks = sam.generate(image_rgb)
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    # Save masks and create gallery
    gallery_images = []
    for i, mask_data in enumerate(masks[:20]):
        mask = mask_data['segmentation']
        
        # Save mask
        mask_uint8 = (mask * 255).astype(np.uint8)
        cv2.imwrite(f"{temp_dir}/{i}.png", mask_uint8)
        
        # Create overlay for gallery
        overlay = image_rgb.copy()
        overlay[mask] = (overlay[mask] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
        cv2.putText(overlay, f"#{i}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        gallery_images.append(overlay)
    
    return gallery_images, temp_dir, f"✅ Found {len(masks)} objects. Enter mask index (0-{len(masks)-1}) below."


def generate_3d(mask_folder, mask_index, seed):
    """Generate 3D model from selected mask"""
    if mask_folder is None or mask_folder == "":
        return None, "❌ Please generate masks first"
    
    try:
        mask_idx = int(mask_index)
    except:
        mask_idx = 0
    
    # Load image and mask
    image_path = f"{mask_folder}/image.png"
    mask_path = f"{mask_folder}/{mask_idx}.png"
    
    if not os.path.exists(image_path):
        return None, "❌ Image not found. Generate masks first."
    if not os.path.exists(mask_path):
        return None, f"❌ Mask {mask_idx} not found. Try a lower index."
    
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 0).astype(np.uint8)
    
    # Run 3D inference
    model = load_3d_model()
    print(f"Generating 3D for mask {mask_idx}...")
    output = model(image, mask, seed=int(seed))
    
    # Save outputs
    output_dir = "gradio_outputs"
    os.makedirs(output_dir, exist_ok=True)
    file_id = str(uuid.uuid4())[:8]
    
    obj_path = f"{output_dir}/model_{file_id}.obj"
    ply_path = f"{output_dir}/model_{file_id}.ply"
    
    output['gs'].save_ply(ply_path)
    
    if 'glb' in output and output['glb'] is not None:
        output['glb'].export(obj_path)
        return obj_path, f"✅ 3D model saved: {obj_path}"
    
    return None, "❌ Failed to generate mesh"


# Build interface
with gr.Blocks(title="SAM 3D Objects") as demo:
    gr.Markdown("# 🎨 SAM 3D Objects - Image to 3D")
    gr.Markdown("**Step 1:** Upload image → **Step 2:** Generate masks → **Step 3:** Select mask → **Step 4:** Generate 3D")
    
    # Hidden state
    mask_folder = gr.State(value=None)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📷 Upload Image")
            image_input = gr.Image(label="Input Image", type="numpy")
            mask_btn = gr.Button("🔍 Step 2: Generate Masks", variant="secondary")
            mask_status = gr.Textbox(label="Mask Status", interactive=False)
        
        with gr.Column():
            gr.Markdown("### 🖼️ Detected Objects")
            gallery = gr.Gallery(label="Masks (green = object)", columns=4, height=300)
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🎯 Generate 3D")
            mask_index = gr.Number(label="Mask Index (0 = largest object)", value=0, precision=0)
            seed_input = gr.Slider(0, 9999, value=42, step=1, label="Seed")
            gen_btn = gr.Button("🚀 Step 4: Generate 3D Model", variant="primary")
            gen_status = gr.Textbox(label="3D Status", interactive=False)
        
        with gr.Column():
            gr.Markdown("### 📦 3D Output")
            output_3d = gr.Model3D(label="3D Model", clear_color=[0.9, 0.9, 0.9, 1.0])
    
    # Events
    mask_btn.click(
        fn=generate_masks,
        inputs=[image_input],
        outputs=[gallery, mask_folder, mask_status]
    )
    
    gen_btn.click(
        fn=generate_3d,
        inputs=[mask_folder, mask_index, seed_input],
        outputs=[output_3d, gen_status]
    )


if __name__ == "__main__":
    os.makedirs("gradio_outputs", exist_ok=True)
    os.makedirs("temp_masks", exist_ok=True)
    
    print("=" * 50)
    print("Pre-loading models (this may take 1-2 minutes)...")
    print("=" * 50)
    load_sam_model()
    load_3d_model()
    print("=" * 50)
    print("Models loaded! Starting server...")
    print("=" * 50)
    
    demo.launch(server_name="0.0.0.0", server_port=7861, share=True)
