#!/usr/bin/env python3
"""
Script to convert 2D images to 3D objects using SAM 3D Objects.
Automatically generates masks using SAM (Segment Anything Model) if masks are not provided.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import argparse

# Import inference utilities
# Try root inference.py first (simpler, fewer dependencies)
try:
    from inference import Inference, load_image
except ImportError:
    # Fallback to notebook version
    notebook_path = os.path.join(os.path.dirname(__file__), 'notebook')
    sys.path.insert(0, notebook_path)
    import inference_notebook as inference_module
    Inference = inference_module.Inference
    load_image = inference_module.load_image

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("Warning: segment_anything not available. Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")


def generate_mask_with_sam(
    image: np.ndarray,
    sam_predictor: Optional['SamPredictor'] = None,
    num_objects: int = 1,
    use_center_point: bool = True
) -> List[np.ndarray]:
    """
    Generate masks for an image using SAM (Segment Anything Model).
    
    Args:
        image: Input image as numpy array (H, W, 3) in RGB format
        sam_predictor: SAM predictor instance (if None, will try to initialize)
        num_objects: Number of objects to segment (default: 1, uses largest mask)
        use_center_point: If True, uses center point prompt; otherwise uses automatic mask generation
    
    Returns:
        List of binary masks
    """
    if not SAM_AVAILABLE:
        raise ImportError("segment_anything is required for automatic mask generation")
    
    # Initialize SAM if not provided
    if sam_predictor is None:
        print("Initializing SAM model...")
        sam_checkpoint = "sam_vit_h_4b8939.pth"  # Default checkpoint
        model_type = "vit_h"
        
        # Try to find SAM checkpoint
        possible_paths = [
            sam_checkpoint,
            f"checkpoints/{sam_checkpoint}",
            f"../{sam_checkpoint}",
            os.path.expanduser(f"~/.cache/sam/{sam_checkpoint}")
        ]
        
        sam_checkpoint_path = None
        for path in possible_paths:
            if os.path.exists(path):
                sam_checkpoint_path = path
                break
        
        if sam_checkpoint_path is None:
            print(f"Warning: SAM checkpoint not found. Please download it from:")
            print("https://github.com/facebookresearch/segment-anything#model-checkpoints")
            print("Trying to use automatic mask generation without checkpoint...")
            # Fallback: create a simple center mask
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            center_y, center_x = h // 2, w // 2
            # Create a mask covering central 60% of image
            mask_size = int(min(h, w) * 0.6)
            y1, y2 = max(0, center_y - mask_size // 2), min(h, center_y + mask_size // 2)
            x1, x2 = max(0, center_x - mask_size // 2), min(w, center_x + mask_size // 2)
            mask[y1:y2, x1:x2] = 1
            return [mask]
        
        try:
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
            sam_predictor = SamPredictor(sam)
        except Exception as e:
            print(f"Error loading SAM model: {e}")
            print("Falling back to simple center mask...")
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            center_y, center_x = h // 2, w // 2
            mask_size = int(min(h, w) * 0.6)
            y1, y2 = max(0, center_y - mask_size // 2), min(h, center_y + mask_size // 2)
            x1, x2 = max(0, center_x - mask_size // 2), min(w, center_x + mask_size // 2)
            mask[y1:y2, x1:x2] = 1
            return [mask]
    
    # Set image for SAM
    sam_predictor.set_image(image)
    
    masks = []
    
    if use_center_point:
        # Use center point as prompt
        h, w = image.shape[:2]
        input_point = np.array([[w // 2, h // 2]])
        input_label = np.array([1])
        
        mask, scores, logits = sam_predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        
        # Select the best mask (highest score)
        best_mask_idx = np.argmax(scores)
        masks.append(mask[best_mask_idx])
    else:
        # Use automatic mask generation
        from segment_anything import SamAutomaticMaskGenerator
        
        mask_generator = SamAutomaticMaskGenerator(sam_predictor.model)
        anns = mask_generator.generate(image)
        
        # Sort by area and take top N
        anns = sorted(anns, key=lambda x: x['area'], reverse=True)
        for ann in anns[:num_objects]:
            masks.append(ann['segmentation'])
    
    return masks


def create_simple_mask(image: np.ndarray, mask_type: str = "center") -> np.ndarray:
    """
    Create a simple mask when SAM is not available.
    
    Args:
        image: Input image
        mask_type: Type of mask ("center", "full", "foreground")
    
    Returns:
        Binary mask
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if mask_type == "center":
        # Center 60% of image
        center_y, center_x = h // 2, w // 2
        mask_size = int(min(h, w) * 0.6)
        y1, y2 = max(0, center_y - mask_size // 2), min(h, center_y + mask_size // 2)
        x1, x2 = max(0, center_x - mask_size // 2), min(w, center_x + mask_size // 2)
        mask[y1:y2, x1:x2] = 1
    elif mask_type == "full":
        # Full image mask
        mask[:] = 1
    elif mask_type == "foreground":
        # Simple foreground detection using color thresholding
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Remove small noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def process_image_to_3d(
    image_path: str,
    output_dir: str,
    mask: Optional[np.ndarray] = None,
    inference: Optional[Inference] = None,
    sam_predictor: Optional = None,
    seed: int = 42,
    use_sam: bool = True
) -> dict:
    """
    Process a single image to 3D object.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs
        mask: Optional pre-computed mask
        inference: SAM 3D inference instance
        sam_predictor: SAM predictor for mask generation
        seed: Random seed for reproducibility
        use_sam: Whether to use SAM for mask generation
    
    Returns:
        Dictionary with output information
    """
    # Load image
    print(f"Loading image: {image_path}")
    image = load_image(image_path)
    
    # Generate mask if not provided
    if mask is None:
        print("Generating mask...")
        if use_sam and SAM_AVAILABLE:
            try:
                masks = generate_mask_with_sam(image, sam_predictor, num_objects=1)
                mask = masks[0]
            except Exception as e:
                print(f"Error generating mask with SAM: {e}")
                print("Falling back to simple center mask...")
                mask = create_simple_mask(image, mask_type="center")
        else:
            mask = create_simple_mask(image, mask_type="center")
    
    # Ensure mask is binary
    mask = (mask > 0).astype(np.uint8)
    
    # Run SAM 3D inference
    print("Running SAM 3D inference...")
    output = inference(image, mask, seed=seed)
    
    # Save outputs
    image_name = Path(image_path).stem
    output_path = os.path.join(output_dir, f"{image_name}_3d.ply")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving 3D model to: {output_path}")
    output["gs"].save_ply(output_path)
    
    # Also save mask for reference
    mask_path = os.path.join(output_dir, f"{image_name}_mask.png")
    cv2.imwrite(mask_path, mask * 255)
    
    return {
        "image_path": image_path,
        "output_path": output_path,
        "mask_path": mask_path,
        "success": True
    }


def main():
    parser = argparse.ArgumentParser(description="Convert 2D images to 3D objects using SAM 3D Objects")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="Images",
        help="Directory containing input images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_3d",
        help="Directory to save 3D outputs"
    )
    parser.add_argument(
        "--checkpoint_tag",
        type=str,
        default="hf",
        help="Checkpoint tag (default: hf)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--use_sam",
        action="store_true",
        default=True,
        help="Use SAM for automatic mask generation"
    )
    parser.add_argument(
        "--no_sam",
        dest="use_sam",
        action="store_false",
        help="Don't use SAM, use simple center mask instead"
    )
    parser.add_argument(
        "--image_extensions",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"],
        help="Image file extensions to process"
    )
    
    args = parser.parse_args()
    
    # Initialize SAM 3D inference
    print("Initializing SAM 3D inference...")
    config_path = f"checkpoints/{args.checkpoint_tag}/pipeline.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Please ensure checkpoints are downloaded. See doc/setup.md for instructions."
        )
    
    inference = Inference(config_path, compile=False)
    
    # Initialize SAM predictor if requested
    sam_predictor = None
    if args.use_sam and SAM_AVAILABLE:
        try:
            print("Initializing SAM predictor...")
            # Try to initialize SAM (will use fallback if checkpoint not found)
            sam_predictor = None  # Will be initialized in generate_mask_with_sam if needed
        except Exception as e:
            print(f"Warning: Could not initialize SAM: {e}")
            print("Will use simple mask generation instead.")
    
    # Find all images
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    image_files = []
    for ext in args.image_extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
    
    if len(image_files) == 0:
        print(f"No images found in {input_dir} with extensions {args.image_extensions}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    results = []
    for i, image_path in enumerate(image_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing image {i}/{len(image_files)}: {image_path.name}")
        print(f"{'='*60}")
        
        try:
            result = process_image_to_3d(
                str(image_path),
                args.output_dir,
                mask=None,
                inference=inference,
                sam_predictor=sam_predictor,
                seed=args.seed,
                use_sam=args.use_sam
            )
            results.append(result)
            print(f"✓ Successfully processed {image_path.name}")
        except Exception as e:
            print(f"✗ Error processing {image_path.name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "image_path": str(image_path),
                "success": False,
                "error": str(e)
            })
    
    # Print summary
    print(f"\n{'='*60}")
    print("Processing Summary")
    print(f"{'='*60}")
    successful = sum(1 for r in results if r.get("success", False))
    print(f"Successfully processed: {successful}/{len(results)}")
    print(f"Output directory: {args.output_dir}")
    
    if successful > 0:
        print("\nGenerated 3D models:")
        for r in results:
            if r.get("success", False):
                print(f"  - {r['output_path']}")


if __name__ == "__main__":
    main()

