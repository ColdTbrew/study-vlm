import json

notebook_path = '/home/seunghyuk/workspace/study_ocr/Vector_search.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# New image_embed function code
new_image_embed_code = [
    "def image_embed(input_data):\n",
    "    # Handle different input types: str (path), PIL Image, or List\n",
    "    if isinstance(input_data, str):\n",
    "        images = PILImage.open(input_data).convert(\"RGB\")\n",
    "    elif isinstance(input_data, PILImage.Image):\n",
    "        images = input_data.convert(\"RGB\")\n",
    "    elif isinstance(input_data, list):\n",
    "        images = []\n",
    "        for item in input_data:\n",
    "            if isinstance(item, str):\n",
    "                images.append(PILImage.open(item).convert(\"RGB\"))\n",
    "            elif isinstance(item, PILImage.Image):\n",
    "                images.append(item.convert(\"RGB\"))\n",
    "    else:\n",
    "        # Fallback, let processor handle or fail\n",
    "        images = input_data\n",
    "\n",
    "    inputs = clip_processor(images=images, return_tensors=\"pt\", padding=True).to(device)\n",
    "    with torch.no_grad():\n",
    "        feats = clip_model.get_image_features(**inputs)\n",
    "    return normalize(feats)\n"
]

# Find the cell defining image_embed
target_cell_index = -1
for i, cell in enumerate(nb['cells']):
    source = "".join(cell['source'])
    if "def image_embed(path: str):" in source:
        target_cell_index = i
        break

if target_cell_index != -1:
    # We need to preserve imports and other functions in that cell if any
    # The cell content was:
    # import torch
    # from PIL import Image as PILImage
    # ...
    # def image_embed(path: str): ...
    # def text_embed(texts): ...
    
    # Let's reconstruct the cell carefully.
    # Actually, looking at the previous view_file, the cell (index 2, execution_count 116) had:
    # imports...
    # clip_id = ...
    # ...
    # def normalize(x): ...
    # def image_embed(path: str): ...
    # def text_embed(texts): ...
    
    original_source = nb['cells'][target_cell_index]['source']
    
    # We will replace the image_embed definition part.
    # It's safer to rewrite the whole cell content with the new function.
    
    new_source = []
    skip = False
    for line in original_source:
        if "def image_embed(path: str):" in line:
            skip = True
            # Insert new function here
            new_source.extend(new_image_embed_code)
        
        if skip:
            # Check if we reached the next function or end of image_embed
            if "def text_embed" in line:
                skip = False
                new_source.append(line)
            # Else, we are skipping the old image_embed body
        else:
            new_source.append(line)
            
    nb['cells'][target_cell_index]['source'] = new_source
    
    print(f"Updated cell {target_cell_index}")
else:
    print("Could not find image_embed definition cell.")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
