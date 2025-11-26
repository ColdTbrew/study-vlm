import json

notebook_path = '/home/seunghyuk/workspace/study_ocr/Vector_search_DINOv3.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 2: Model Loading (Ensure it matches user's preference)
# Note: The user's snippet uses device_map="auto", so we don't need to manually .to(device) the model
new_model_loading_code = [
    "# 2. DINOv3 모델 로드\n",
    "from transformers import AutoImageProcessor, AutoModel\n",
    "\n",
    "pretrained_model_name = \"facebook/dinov3-vitl16-pretrain-lvd1689m\"\n",
    "print(f\"Loading model: {pretrained_model_name}...\")\n",
    "\n",
    "processor = AutoImageProcessor.from_pretrained(pretrained_model_name)\n",
    "model = AutoModel.from_pretrained(\n",
    "    pretrained_model_name, \n",
    "    device_map=\"auto\", \n",
    ")\n",
    "print(\"Model loaded successfully.\")"
]

# Find the cell that loads the model. It was index 2 (0: markdown, 1: imports, 2: login, 3: model load).
# Wait, I inserted a login cell at index 2 previously. So Model Loading is now index 3.
# Let's check the content to be sure.
model_load_cell_index = -1
for i, cell in enumerate(nb['cells']):
    if "DINOv3 모델 로드" in "".join(cell['source']):
        model_load_cell_index = i
        break

if model_load_cell_index != -1:
    nb['cells'][model_load_cell_index]['source'] = new_model_loading_code

# Update Cell 4 (was 3): Embedding Function
# Simplify: No try-except, direct pooler_output usage
new_embedding_code = [
    "# 3. 임베딩 추출 함수 정의\n",
    "def get_embedding(image_path):\n",
    "    # 예외 처리 제거 및 코드 간소화\n",
    "    image = PILImage.open(image_path).convert(\"RGB\")\n",
    "    inputs = processor(images=image, return_tensors=\"pt\").to(model.device)\n",
    "    \n",
    "    with torch.inference_mode():\n",
    "        outputs = model(**inputs)\n",
    "        \n",
    "    embedding = outputs.pooler_output\n",
    "    \n",
    "    # Normalize (L2) for Cosine Similarity\n",
    "    embedding = embedding / embedding.norm(dim=-1, keepdim=True)\n",
    "    return embedding.cpu()"
]

embedding_cell_index = -1
for i, cell in enumerate(nb['cells']):
    if "임베딩 추출 함수 정의" in "".join(cell['source']):
        embedding_cell_index = i
        break

if embedding_cell_index != -1:
    nb['cells'][embedding_cell_index]['source'] = new_embedding_code

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
