import sys
sys.path.append('notebook')
from inference import Inference, load_image, load_single_mask

# Load model
tag = 'hf'
config_path = f'checkpoints/{tag}/pipeline.yaml'
inference = Inference(config_path, compile=False)

# Load image and mask
image = load_image('notebook/images/testimages/test2.jpg')
mask = load_single_mask('/data0/ram_codes/sam-3d-objects/my_output', index=12)

# Run inference
output = inference(image, mask, seed=42)

# Export results
# Save Gaussian Splat as PLY
output['gs'].save_ply('splat.ply')
print("Gaussian Splat saved to splat.ply")

# Save Mesh as OBJ
if 'glb' in output and output['glb'] is not None:
    output['glb'].export('output.obj')
    print("Mesh saved to output.obj")
else:
    print("No mesh output available")
