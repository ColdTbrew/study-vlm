import os
import pickle
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sam3.model_builder import build_sam3_image_model
from sam3_utils import CustomSam3Processor
from transformers import AutoImageProcessor, AutoModel

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize SAM3
bpe_path = os.path.join("assets", "bpe_simple_vocab_16e6.txt.gz")
if not os.path.exists(bpe_path):
    bpe_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "bpe_simple_vocab_16e6.txt.gz")

print(f"Using BPE path: {bpe_path}")

try:
    sam3_model = build_sam3_image_model(bpe_path=bpe_path).to(device)
    sam3_processor = CustomSam3Processor(sam3_model, confidence_threshold=0.3)
except Exception as e:
    print(f"Failed to load SAM3 model: {e}")
    exit(1)

# Initialize DINOv3
dino_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"
try:
    dino_processor = AutoImageProcessor.from_pretrained(dino_id)
    dino_model = AutoModel.from_pretrained(dino_id).to(device)
    dino_model.eval()
except Exception as e:
    print(f"Failed to load DINOv3 model: {e}")
    exit(1)

def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)

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

def crop_by_mask(image: Image.Image, mask: np.ndarray, pad: int = 4):
    y_indices, x_indices = np.where(mask > 0.5)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None

    x1, x2 = x_indices.min(), x_indices.max()
    y1, y2 = y_indices.min(), y_indices.max()

    x1 = max(x1 - pad, 0)
    y1 = max(y1 - pad, 0)
    x2 = min(x2 + pad, image.width - 1)
    y2 = min(y2 + pad, image.height - 1)

    return image.crop((x1, y1, x2 + 1, y2 + 1))

# Main Loop
image_dir = "../data/images_original"
if not os.path.exists(image_dir):
    # Try absolute path
    image_dir = "/home/seunghyuk/workspace/study_ocr/data/images_original"

print(f"Scanning images in {image_dir}")
image_files = [
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

vector_store = []
print(f"Indexing {len(image_files)} images (single-image loop)...")
fail_stats = {"mask_none": 0, "crop_none": 0, "exception": 0}

# Create output directory for failed visualizations
vis_dir = "failed_mask_visualizations"
os.makedirs(vis_dir, exist_ok=True)

for path in tqdm(image_files):
    try:
        image = Image.open(path).convert("RGB")
        mask = segment_clothes(image, prompt="segment anything center of image")
        if mask is None:
            fail_stats["mask_none"] += 1
            print(f"Skip mask_none: {path}")
            
            # Visualization
            plt.figure(figsize=(5, 5))
            plt.imshow(image)
            plt.title(f"Mask None: {os.path.basename(path)}")
            plt.axis("off")
            
            # Save visualization to file since we are running in script
            vis_path = os.path.join(vis_dir, f"fail_mask_{os.path.basename(path)}")
            plt.savefig(vis_path)
            plt.close()
            print(f"Saved visualization to {vis_path}")
            
            continue

        masked_image = apply_mask_with_gray(image, mask)
        crop = crop_by_mask(masked_image, mask)
        if crop is None:
            fail_stats["crop_none"] += 1
            print(f"Skip crop_none: {path}")
            continue

        inputs = dino_processor(images=crop, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = dino_model(**inputs)
            cls = outputs.last_hidden_state[:, 0]
            emb = l2_normalize(cls).cpu()
        vector_store.append({"path": path, "embedding": emb})
    except Exception as e:
        fail_stats["exception"] += 1
        print(f"Skip exception: {path} -> {e}")

print("Fails:", fail_stats)

print(f"Indexed {len(vector_store)} images.")
output_path = "vector_store_sam3_bg_gray_dino.pkl"
with open(output_path, "wb") as f:
    pickle.dump(vector_store, f)
print(f"Vector store saved to {output_path}")
