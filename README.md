# study-ocr

Jupyter 실습용 OCR/VLM 연습 프로젝트입니다. `uv`로 의존성을 관리하고, 노트북(`ocr_practice.ipynb`)에서 LLaVA, BLIP-2 Q-Former, CLIP을 순서대로 체험합니다.

## Quickstart

```bash
uv venv .venv
source .venv/bin/activate
# CUDA 사용 시:
UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121 uv sync
# CPU-only:
uv sync

python -m ipykernel install --user --name study-ocr
jupyter lab ocr_practice.ipynb
```

## CLIP 시각화 (어떤 부분을 보는지)

Vision Transformer의 CLS 토큰이 어떤 패치에 주목했는지 히트맵으로 그립니다.

```bash
uv run python visualize_clip_attention.py \
  --image sample_images/cat-dog.png \
  --model openai/clip-vit-large-patch14 \
  --output clip_attention_overlay.png
```

`clip_attention_overlay.png`에 원본과 히트맵이 나란히 저장됩니다.
