import torch
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from sam3.model_builder import build_sam3_image_model
from sam3_utils import CustomSam3Processor

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize SAM3
# Try to locate bpe file
bpe_path = os.path.join("assets", "bpe_simple_vocab_16e6.txt.gz")
if not os.path.exists(bpe_path):
    # Try absolute path relative to this script
    bpe_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "bpe_simple_vocab_16e6.txt.gz")

print(f"Using BPE path: {bpe_path}")

try:
    sam3_model = build_sam3_image_model(bpe_path=bpe_path).to(device)
    sam3_processor = CustomSam3Processor(sam3_model, confidence_threshold=0.3)
except Exception as e:
    print(f"Failed to load SAM3 model: {e}")
    exit(1)

def segment_clothes(image: Image.Image, prompt: str = "clothing"):
    inference_state = sam3_processor.set_image(image)
    out = sam3_processor.set_text_prompt(state=inference_state, prompt=prompt)
    
    masks = out["masks"]
    scores = out["scores"]

    if masks is None or len(masks) == 0:
        return None

    masks = masks.to(device)
    scores = scores.to(device)

    # Filter by score
    keep = scores > 0.3
    if keep.any():
        sel_masks = masks[keep]
    else:
        idx = torch.argmax(scores)
        sel_masks = masks[idx : idx + 1]

    # Select best mask based on Area and Center
    if sel_masks.shape[0] > 1:
        # Ensure float for calculation
        sel_masks_float = sel_masks.float()
        
        # Calculate Area
        areas = sel_masks_float.sum(dim=(1, 2))
        
        # Calculate Centroid
        H, W = sel_masks.shape[1], sel_masks.shape[2]
        
        # Grid for centroids
        y_grid = torch.arange(H, device=device).view(-1, 1).expand(H, W)
        x_grid = torch.arange(W, device=device).view(1, -1).expand(H, W)
        
        total_mass = areas.clone()
        total_mass[total_mass == 0] = 1.0
        
        cen_y = (sel_masks_float * y_grid).sum(dim=(1, 2)) / total_mass
        cen_x = (sel_masks_float * x_grid).sum(dim=(1, 2)) / total_mass
        
        center_y, center_x = H / 2.0, W / 2.0
        dists = torch.sqrt((cen_y - center_y)**2 + (cen_x - center_x)**2)
        max_dist = (H**2 + W**2)**0.5 / 2.0
        
        # Normalize metrics
        norm_area = areas / (areas.max() + 1e-6)
        norm_dist = 1.0 - (dists / max_dist)
        
        # Combined Score: 0.4 Area + 0.6 Center
        combined_score = 0.4 * norm_area + 0.6 * norm_dist
        
        best_idx = torch.argmax(combined_score)
        final_mask = sel_masks[best_idx]
    else:
        final_mask = sel_masks[0]

    final_mask = final_mask.float()
    final_mask = final_mask.cpu().numpy()
    
    if final_mask.ndim == 3 and final_mask.shape[0] == 1:
        final_mask = final_mask.squeeze(0)
        
    return final_mask

def apply_mask_with_gray(image: Image.Image, mask: np.ndarray, gray_value: int = 128):
    img_arr = np.array(image)
    gray_bg = np.full_like(img_arr, fill_value=gray_value, dtype=np.uint8)

    if mask.shape != img_arr.shape[:2]:
        from PIL import Image as PilImage
        mask_img = PilImage.fromarray((mask * 255).astype(np.uint8))
        mask_img = mask_img.resize((img_arr.shape[1], img_arr.shape[0]), PilImage.NEAREST)
        mask = np.array(mask_img).astype(np.float32) / 255.0

    mask_3d = np.stack([mask] * 3, axis=-1)
    binary_mask = mask_3d > 0.5
    result_arr = np.where(binary_mask, img_arr, gray_bg)

    return Image.fromarray(result_arr)

# Main execution
img_path = "../data/images_original/e5d94424-9941-4eb7-985a-a68b85bf3781.jpg"

if not os.path.exists(img_path):
    # Try absolute path based on workspace
    img_path = "/home/seunghyuk/workspace/study_ocr/data/images_original/e5d94424-9941-4eb7-985a-a68b85bf3781.jpg"

print(f"Processing image: {img_path}")
if not os.path.exists(img_path):
    print("Image file does not exist.")
    exit(1)

try:
    img = Image.open(img_path).convert("RGB")
    mask = segment_clothes(img, prompt="clothing")

    if mask is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("SAM Mask")
        axes[1].axis("off")
        
        masked_img = apply_mask_with_gray(img, mask)
        axes[2].imshow(masked_img)
        axes[2].set_title("Gray Background")
        axes[2].axis("off")
        
        output_path = "sam_visualization.png"
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    else:
        print("No mask detected.")
except Exception as e:
    print(f"Error during processing: {e}")
    import traceback
    traceback.print_exc()
