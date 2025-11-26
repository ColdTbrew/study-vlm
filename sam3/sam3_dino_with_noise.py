import os
import sys
import pickle
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt  # 디버깅용, 지금은 안 쓰지만 놔둠

# Revert to using sam3 package as transformers support is not ready
from sam3.model_builder import build_sam3_image_model
# Use CustomSam3Processor for batch support
try:
    from sam3.sam3_utils import CustomSam3Processor
except ImportError:
    # Fallback
    from sam3_utils import CustomSam3Processor

from transformers import AutoImageProcessor, AutoModel
from torch.utils.data import DataLoader

# -----------------------------
# Device 설정
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -----------------------------
# SAM3 초기화 (BPE 경로 처리)
# -----------------------------
# 기본 상대경로
bpe_path = os.path.join("assets", "bpe_simple_vocab_16e6.txt.gz")

if not os.path.exists(bpe_path):
    # sam3 패키지 구조 안에서 실행할 때를 위한 fallback
    bpe_path_alt = os.path.join(os.getcwd(), "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
    if os.path.exists(bpe_path_alt):
        bpe_path = bpe_path_alt

if not os.path.exists(bpe_path):
    # 경고만 찍고 진행하면 내부에서 다시 FileNotFoundError 날 수 있으니 명시적으로 죽이기
    raise FileNotFoundError(
        f"BPE path not found. Tried: {bpe_path}. "
        "Please make sure 'bpe_simple_vocab_16e6.txt.gz' exists under 'assets/' or 'sam3/assets/'."
    )

print(f"Using BPE path: {bpe_path}")

sam3_model = build_sam3_image_model(bpe_path=bpe_path).to(device)
# Use CustomSam3Processor
sam3_processor = CustomSam3Processor(sam3_model, confidence_threshold=0.3)

# -----------------------------
# DINOv3 초기화
# -----------------------------
dino_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"
dino_processor = AutoImageProcessor.from_pretrained(dino_id)
dino_model = AutoModel.from_pretrained(dino_id).to(device)
dino_model.eval()


# -----------------------------
# Utility 함수
# -----------------------------
def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)


def resize_mask_to_image(mask: np.ndarray, image_shape_hw):
    """
    mask (Hm, Wm)을 이미지 shape (H, W)에 맞게 리사이즈.
    image_shape_hw: (H, W)
    """
    h, w = image_shape_hw
    if mask.shape == (h, w):
        return mask

    # mask는 0~1 float이라고 가정
    from PIL import Image as PilImage

    mask_img = PilImage.fromarray((mask * 255).astype(np.uint8))
    mask_img = mask_img.resize((w, h), PilImage.NEAREST)
    resized = np.array(mask_img).astype(np.float32) / 255.0
    return resized


def apply_mask_with_noise(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """
    이미지에서 mask 영역은 원본을 유지하고, 나머지는 Gaussian noise로 채움.
    mask: (H, W) float [0,1] 혹은 bool
    """
    img_arr = np.array(image)
    h, w = img_arr.shape[:2]

    # mask size가 안 맞으면 리사이즈
    if mask.shape != (h, w):
        mask = resize_mask_to_image(mask, (h, w))

    # noise 배경 생성
    noise = np.random.normal(loc=128, scale=30, size=img_arr.shape).astype(np.uint8)

    # 3채널로 확장
    mask_3d = np.stack([mask] * 3, axis=-1)
    binary_mask = mask_3d > 0.5

    result_arr = np.where(binary_mask, img_arr, noise)
    return Image.fromarray(result_arr)


def crop_by_mask(image: Image.Image, mask: np.ndarray, pad: int = 4):
    """
    mask > 0.5인 영역을 bounding box로 잡고 pad만큼 여유를 두어 crop.
    """
    h, w = image.size[1], image.size[0]

    # mask size 맞추기
    if mask.shape != (h, w):
        mask = resize_mask_to_image(mask, (h, w))

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


# -----------------------------
# 단일 이미지용 세그멘테이션 & 임베딩
# -----------------------------
def segment_clothes(image: Image.Image, prompt: str = "clothing"):
    """
    단일 이미지에 대해 SAM3로 clothing segment를 얻고
    binary mask (H, W)를 np.ndarray로 반환.
    """
    inference_state = sam3_processor.set_image(image)
    out = sam3_processor.set_text_prompt(state=inference_state, prompt=prompt)

    masks = out["masks"]
    scores = out["scores"]

    if masks is None or scores is None or len(masks) == 0:
        return None

    masks = masks.to(device)
    scores = scores.to(device)

    # score 기준 필터링
    keep = scores > 0.3
    if keep.any():
        sel_masks = masks[keep]
    else:
        idx = torch.argmax(scores)
        sel_masks = masks[idx: idx + 1]

    # [N, H, W] 아닐 경우를 대비
    if sel_masks.dim() == 2:
        sel_masks = sel_masks.unsqueeze(0)

    merged = sel_masks.any(dim=0).float()
    merged = merged.cpu().numpy()

    # [1, H, W] 형태일 경우 squeeze
    if merged.ndim == 3 and merged.shape[0] == 1:
        merged = merged.squeeze(0)

    return merged


def clothes_embedding(image_path: str, prompt: str = "clothing") -> torch.Tensor | None:
    """
    단일 이미지 파일 경로에 대해:
    - clothing segment
    - 노이즈 배경 적용
    - crop
    - DINOv3 CLS 임베딩 반환 (shape: [1, D]) 또는 None
    """
    try:
        image = Image.open(image_path).convert("RGB")
        mask = segment_clothes(image, prompt=prompt)

        if mask is None:
            return None

        masked_image = apply_mask_with_noise(image, mask)
        crop = crop_by_mask(masked_image, mask)

        if crop is None:
            return None

        inputs = dino_processor(images=crop, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = dino_model(**inputs)
            cls = outputs.last_hidden_state[:, 0]
            emb = l2_normalize(cls)

        return emb.cpu()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


# -----------------------------
# DataLoader / Dataset import
# -----------------------------
# sam3/dataset_utils.py 를 불러오기 위한 path 설정
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

try:
    from sam3.dataset_utils import ImageDataset, collate_fn
except ImportError:
    # sam3 디렉토리 안에서 바로 실행할 때
    from dataset_utils import ImageDataset, collate_fn


# -----------------------------
# 배치 세그멘테이션
# -----------------------------
def segment_clothes_batch(images: list[Image.Image], prompt: str = "clothing"):
    """
    여러 이미지를 한 번에 SAM3에 넣어서
    masks, scores를 반환.
    SAM3의 set_image_batch가 state를 리턴한다고 가정하고 사용.
    """
    # state={}를 넘기고 반환값을 받는 형태로 통일 (single과 패턴 맞추기)
    inference_state = sam3_processor.set_image_batch(images, state={})
    
    # Use CustomSam3Processor's batch method
    prompts = [prompt] * len(images)
    out = sam3_processor.set_text_prompt_batch(prompts=prompts, state=inference_state)

    return out["masks"], out["scores"]


def process_batch(images, paths, prompt: str = "clothing"):
    """
    한 배치에 대해:
    - SAM3 batch 세그멘테이션
    - 각 이미지에 대해 마스크 선택/머지 → 노이즈 배경 → crop
    - DINOv3 batch 임베딩
    - [{"path": str, "embedding": Tensor}, ...] 리스트 반환
    """
    if not images:
        return []

    try:
        masks_batch, scores_batch = segment_clothes_batch(images, prompt=prompt)

        if masks_batch is None or scores_batch is None:
            return []

        crops = []
        valid_indices = []

        for i, (image, masks, scores) in enumerate(zip(images, masks_batch, scores_batch)):
            try:
                if masks is None or scores is None or scores.numel() == 0:
                    continue

                # [N, H, W] 혹은 [H, W]를 [N, H, W] 형태로 맞춤
                if masks.dim() == 2:
                    masks = masks.unsqueeze(0)

                # score 기준 필터링
                keep = scores > 0.3
                if keep.any():
                    sel_masks = masks[keep]
                else:
                    idx = torch.argmax(scores)
                    sel_masks = masks[idx: idx + 1]

                if sel_masks.dim() == 2:
                    sel_masks = sel_masks.unsqueeze(0)

                merged = sel_masks.any(dim=0).float()
                merged = merged.cpu().numpy()

                if merged.ndim == 3 and merged.shape[0] == 1:
                    merged = merged.squeeze(0)

                # 노이즈 배경 적용
                masked_image = apply_mask_with_noise(image, merged)

                # crop
                crop = crop_by_mask(masked_image, merged)
                if crop is not None:
                    crops.append(crop)
                    valid_indices.append(i)

            except Exception as e:
                print(f"Error processing mask for {paths[i]}: {e}")
                continue

        if not crops:
            return []

        # DINOv3 batch inference
        inputs = dino_processor(images=crops, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = dino_model(**inputs)
            cls_tokens = outputs.last_hidden_state[:, 0]
            embeddings = l2_normalize(cls_tokens).cpu()

        results = []
        for idx, emb in zip(valid_indices, embeddings):
            # 필요에 따라 numpy로 바꾸고 싶으면 emb.numpy() 사용
            results.append({
                "path": paths[idx],
                "embedding": emb
            })

        return results

    except Exception as e:
        print(f"Batch processing error: {e}")
        return []


# -----------------------------
# Indexing 파이프라인
# -----------------------------
if __name__ == "__main__":
    image_dir = "../data/images_original"
    image_files = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    batch_size = 64  # VRAM 상황에 따라 조정
    num_workers = 1  # Jupyter에서 테스트할 땐 0으로 두는 게 안전

    dataset = ImageDataset(image_files)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    vector_store = []

    print(f"Indexing {len(image_files)} images with batch size {batch_size}...")
    for images, paths in tqdm(dataloader):
        batch_results = process_batch(images, paths)
        vector_store.extend(batch_results)

    print(f"Indexed {len(vector_store)} images.")

    # Save vector store
    output_path = "vector_store_sam3_bg_noise_dino.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(vector_store, f)

    print(f"Vector store saved to {output_path}")