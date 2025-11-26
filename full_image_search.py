import os
import torch
from PIL import Image as PILImage
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

# 1. Load Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

def get_image_embedding(image_path):
    try:
        image = PILImage.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        return image_features / image_features.norm(dim=-1, keepdim=True)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def get_text_embedding(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    return text_features / text_features.norm(dim=-1, keepdim=True)

# 2. Indexing All Images
image_folder = "./data/images_original"
db_path = "image_embeddings.pkl"

if os.path.exists(db_path):
    print("Loading existing database...")
    with open(db_path, 'rb') as f:
        db = pickle.load(f)
else:
    print("Indexing images...")
    db = {"paths": [], "embeddings": []}
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for filename in tqdm(image_files):
        path = os.path.join(image_folder, filename)
        emb = get_image_embedding(path)
        if emb is not None:
            db["paths"].append(path)
            db["embeddings"].append(emb.cpu())
    
    # Concatenate all embeddings into a single tensor
    if db["embeddings"]:
        db["embeddings"] = torch.cat(db["embeddings"], dim=0)
        
    with open(db_path, 'wb') as f:
        pickle.dump(db, f)
    print(f"Indexed {len(db['paths'])} images.")

# 3. Search Function
def search(query, top_k=5):
    # Text Embedding
    text_emb = get_text_embedding(query).cpu()
    
    # Similarity
    # db["embeddings"]: [N, D], text_emb: [1, D] -> [D, 1]
    sims = (db["embeddings"] @ text_emb.T).squeeze()
    
    # Top K
    values, indices = torch.topk(sims, top_k)
    
    print(f"Query: '{query}'")
    
    # Visualization
    fig, axes = plt.subplots(1, top_k, figsize=(5 * top_k, 5))
    if top_k == 1: axes = [axes]
        
    for i, idx in enumerate(indices):
        img_path = db["paths"][idx.item()]
        score = values[i].item()
        
        img = PILImage.open(img_path)
        ax = axes[i]
        ax.imshow(img)
        ax.set_title(f"Score: {score:.4f}\n{os.path.basename(img_path)}")
        ax.axis('off')
    plt.show()

# Example Usage
# search("a photo of a cat")
