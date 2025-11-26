import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PilImage


def visualize_noise(image: PilImage.Image, mask: np.ndarray, save_path: str | None = None):
    img_arr = np.array(image)
    h, w = img_arr.shape[:2]

    # 마스크 크기 보정
    if mask.shape != (h, w):
        mask = np.array(PilImage.fromarray((mask * 255).astype(np.uint8)).resize((w, h), PilImage.NEAREST)) / 255.0

    # 노이즈 생성
    noise = np.random.normal(loc=128, scale=30, size=img_arr.shape).astype(np.uint8)

    mask_3d = np.stack([mask] * 3, axis=-1)
    binary_mask = mask_3d > 0.5
    result_arr = np.where(binary_mask, img_arr, noise)

    # 저장 옵션
    if save_path:
        PilImage.fromarray(result_arr).save(save_path, format="JPEG")

    # 시각화
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img_arr); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(mask, cmap="gray"); axes[1].set_title("Mask"); axes[1].axis("off")
    axes[2].imshow(noise); axes[2].set_title("Gaussian Noise"); axes[2].axis("off")
    axes[3].imshow(result_arr); axes[3].set_title("Masked w/ Noise"); axes[3].axis("off")
    plt.tight_layout()
    plt.show()
def apply_mask_with_noise(image, mask, invert=False):
    mask = resize_mask_to_image(mask, image.size[::-1])
    mask_3d = np.stack([mask] * 3, axis=-1)
    if invert:
        mask_3d = 1.0 - mask_3d
    binary_mask = mask_3d > 0.5
    noise = np.random.normal(loc=128, scale=30, size=np.array(image).shape).astype(np.uint8)
    return Image.fromarray(np.where(binary_mask, np.array(image), noise))


def resize_mask_to_image(mask: np.ndarray, image_hw):
    h, w = image_hw
    if mask.shape == (h, w):
        return mask
    from PIL import Image as PilImage
    m = PilImage.fromarray((mask * 255).astype(np.uint8)).resize((w, h), PilImage.NEAREST)
    return np.array(m).astype(np.float32) / 255.0

if __name__ == "__main__":
    # 사용 예시
    image_path = "./samples/1.png"
    mask_path = "./samples/1_mask.png"  # 별도 마스크가 있으면 사용, 없으면 이미지 그레이스케일을 마스크로 사용

    image = PilImage.open(image_path).convert("RGB")

    if os.path.exists(mask_path):
        mask = np.array(PilImage.open(mask_path)).astype(np.float32) / 255.0
    else:
        mask = np.array(image.convert("L")).astype(np.float32) / 255.0

    out_path = "./samples/1_masked_noise.jpg"
    visualize_noise(image, mask, save_path=out_path)
    print(f"Saved masked image to: {out_path}")
    import os, numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    img_path = "./samples/1.png"
    mask_path = "./samples/1_mask.png"

    image = Image.open(img_path).convert("RGB")
    mask = np.array(Image.open(mask_path)).astype(np.float32) / 255.0 if os.path.exists(mask_path) \
        else (np.array(image.convert("L")).astype(np.float32) / 255.0 > 0.5).astype(np.float32)

    masked = apply_mask_with_noise(image, mask, invert=True)

    # 결과 이미지 자체 저장
    masked.save("masked_noise.jpg")

    # 비교 시각화 저장
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(mask, cmap="gray"); axes[1].set_title("Mask"); axes[1].axis("off")
    axes[2].imshow(masked); axes[2].set_title("Masked w/ Noise"); axes[2].axis("off")
    plt.tight_layout()
    fig.savefig("masked_noise_vis.jpg", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved masked image to masked_noise.jpg and visualization to masked_noise_vis.jpg")
