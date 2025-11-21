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
