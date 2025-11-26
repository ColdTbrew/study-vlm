import torch
import numpy as np
from PIL import Image
import os
# Revert to using sam3 package as transformers support is not ready
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from transformers import AutoImageProcessor, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize SAM3

bpe_path = os.path.join(os.path.dirname(__file__), "assets", "bpe_simple_vocab_16e6.txt.gz")
sam3_model = build_sam3_image_model(bpe_path=bpe_path).to(device)
sam3_processor = Sam3Processor(sam3_model, confidence_threshold=0.3)

# Initialize DINOv3
dino_id = "facebook/dinov3-vitl16-pretrain-lvd1689m"
dino_processor = AutoImageProcessor.from_pretrained(dino_id)
dino_model = AutoModel.from_pretrained(dino_id).to(device)
dino_model.eval()

def get_embedding(image_path):
    image = PILImage.open(image_path).convert("RGB")
    inputs = dino_processor(images=image, return_tensors="pt").to(dino_model.device)
    
    with torch.inference_mode():
        outputs = dino_model(**inputs)
        
    # Use pooler_output as requested
    # Note: Some models might not have pooler_output, in that case we might need a fallback
    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
        embedding = outputs.pooler_output
    else:
        # Fallback to CLS token if pooler_output is not available (common in some ViT implementations)
        embedding = outputs.last_hidden_state[:, 0, :]
        
    # Normalize (L2)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu()


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)


def segment_clothes(image: Image.Image, prompt: str = "clothing"):
    # SAM3 processor usage in sam3 package is different from transformers
    inference_state = sam3_processor.set_image(image)
    out = sam3_processor.set_text_prompt(state=inference_state, prompt=prompt)
    
    masks = out["masks"]
    scores = out["scores"]

    if masks is None or len(masks) == 0:
        raise RuntimeError("No masks returned from SAM3.")

    masks = masks.to(device)
    scores = scores.to(device)

    # Filter by score
    keep = scores > 0.3
    if keep.any():
        sel_masks = masks[keep]
    else:
        idx = torch.argmax(scores)
        sel_masks = masks[idx : idx + 1]

    # Merge masks
    # Merge masks
    merged = sel_masks.any(dim=0).float()
    merged = merged.cpu().numpy()
    
    print(f"DEBUG: merged mask shape before squeeze: {merged.shape}")
    if merged.ndim == 3 and merged.shape[0] == 1:
        merged = merged.squeeze(0)
    print(f"DEBUG: merged mask shape after squeeze: {merged.shape}")
        
    return merged


def crop_by_mask(image: Image.Image, mask: np.ndarray, pad: int = 4):
    y_indices, x_indices = np.where(mask > 0.5)
    if len(x_indices) == 0 or len(y_indices) == 0:
        raise RuntimeError("Empty mask after thresholding.")

    x1, x2 = x_indices.min(), x_indices.max()
    y1, y2 = y_indices.min(), y_indices.max()

    x1 = max(x1 - pad, 0)
    y1 = max(y1 - pad, 0)
    x2 = min(x2 + pad, image.width - 1)
    y2 = min(y2 + pad, image.height - 1)

    return image.crop((x1, y1, x2 + 1, y2 + 1))


def clothes_embedding(image_path: str, prompt: str = "clothing") -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    mask = segment_clothes(image, prompt=prompt)

    # Save intermediate mask
    mask_vis = (mask * 255).astype(np.uint8)
    Image.fromarray(mask_vis).save("intermediate_mask.jpg")

    crop = crop_by_mask(image, mask)

    # Save intermediate crop
    crop.save("intermediate_crop.jpg")

    inputs = dino_processor(images=crop, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino_model(**inputs)
        cls = outputs.last_hidden_state[:, 0]
        emb = l2_normalize(cls)

    return emb  # shape: [1, D]


if __name__ == "__main__":
    emb = clothes_embedding("../sample_images_from_kaggle/1.jpg", prompt="clothing")
    print(emb.shape, emb.norm(dim=-1))