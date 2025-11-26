import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def compute_cls_attention_map(model: CLIPModel, pixel_values: torch.Tensor) -> torch.Tensor:
    """Return a single attention heatmap (B, H, W) based on the CLS token from the last layer."""
    with torch.no_grad():
        vision_out = model.vision_model(
            pixel_values=pixel_values,
            output_attentions=True,
            return_dict=True,
        )

    # Use the last layer, average heads, then take CLS -> patch attention
    last_attn = vision_out.attentions[-1]  # (batch, heads, seq, seq)
    mean_attn = last_attn.mean(dim=1)  # (batch, seq, seq)
    cls_to_patches = mean_attn[:, 0, 1:]  # drop CLS-to-CLS

    num_patches = cls_to_patches.shape[-1]
    grid_size = int(math.isqrt(num_patches))
    if grid_size * grid_size != num_patches:
        raise ValueError(f"Unexpected patch count: {num_patches}")

    coarse_map = cls_to_patches.reshape(-1, 1, grid_size, grid_size)
    return F.interpolate(
        coarse_map,
        size=pixel_values.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


def save_overlay(image: Image.Image, attn_map: torch.Tensor, out_path: Path) -> None:
    attn = attn_map[0].detach().cpu().numpy()
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
    attn_img = Image.fromarray((attn * 255).astype("uint8")).resize(image.size, Image.BILINEAR)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(image)
    axes[1].imshow(attn_img, cmap="magma", alpha=0.5)
    axes[1].set_title("CLIP focus (CLS → patches)")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved overlay to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize which regions CLIP attends to for its image embedding.")
    parser.add_argument("--image", type=str, default="test_image.jpg", help="Path to an RGB image.")
    parser.add_argument(
        "--model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="CLIP vision backbone to use.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="clip_attention_overlay.png",
        help="Output path for the overlay plot.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    image = Image.open(args.image).convert("RGB")
    processor = CLIPProcessor.from_pretrained(args.model)
    inputs = processor(images=image, return_tensors="pt").to(device)

    model = CLIPModel.from_pretrained(args.model).to(device)
    model.eval()
    attn_map = compute_cls_attention_map(model, inputs["pixel_values"])
    save_overlay(image, attn_map, Path(args.output))


if __name__ == "__main__":
    main()
