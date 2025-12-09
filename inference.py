import cv2
import numpy as np
from omegaconf import OmegaConf
from hydra.utils import instantiate
from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot load image: {path}")
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_single_mask(folder, index=0):
    mask_path = f"{folder}/{index}.png"
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot load mask: {mask_path}")
    mask = (mask > 0).astype(np.uint8)
    return mask

class Inference:
    def __init__(self, config_file, compile=False):
        cfg = OmegaConf.load(config_file)
        cfg.rendering_engine = "pytorch3d"
        cfg.compile_model = compile
        cfg.workspace_dir = config_file.rsplit("/", 1)[0]
        
        # Instantiate the full pipeline from config (includes depth_model)
        self.pipeline = instantiate(cfg)

    def merge_mask(self, img, mask):
        mask = (mask[..., None] * 255).astype(np.uint8)
        return np.concatenate([img[..., :3], mask], axis=-1)

    def __call__(self, image, mask=None, seed=42):
        rgba = self.merge_mask(image, mask)
        return self.pipeline.run(
            rgba,
            None,
            seed,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=True,
            use_vertex_color=True,
        )
