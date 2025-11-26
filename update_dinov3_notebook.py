import json

notebook_path = '/home/seunghyuk/workspace/study_ocr/Vector_search_DINOv3.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update the model loading cell (Cell 2)
# We'll replace the try-except block with the user's provided code structure
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

nb['cells'][2]['source'] = new_model_loading_code

# Update the embedding extraction function (Cell 3) to use pooler_output as shown in user code
new_embedding_code = [
    "# 3. 임베딩 추출 함수 정의\n",
    "def get_embedding(image_path):\n",
    "    try:\n",
    "        image = PILImage.open(image_path).convert(\"RGB\")\n",
    "        inputs = processor(images=image, return_tensors=\"pt\").to(model.device)\n",
    "        \n",
    "        with torch.inference_mode():\n",
    "            outputs = model(**inputs)\n",
    "            \n",
    "        # Use pooler_output as requested\n",
    "        # Note: Some models might not have pooler_output, in that case we might need a fallback\n",
    "        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:\n",
    "            embedding = outputs.pooler_output\n",
    "        else:\n",
    "            # Fallback to CLS token if pooler_output is not available (common in some ViT implementations)\n",
    "            embedding = outputs.last_hidden_state[:, 0, :]\n",
    "            \n",
    "        # Normalize (L2)\n",
    "        embedding = embedding / embedding.norm(dim=-1, keepdim=True)\n",
    "        return embedding.cpu()\n",
    "    except Exception as e:\n",
    "        print(f\"Error processing {image_path}: {e}\")\n",
    "        return None"
]

nb['cells'][3]['source'] = new_embedding_code

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
