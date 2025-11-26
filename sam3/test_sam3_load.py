from transformers import AutoModel, AutoProcessor
import torch

try:
    model = AutoModel.from_pretrained("facebook/sam3", trust_remote_code=True)
    print("Model loaded successfully")
    print(type(model))
except Exception as e:
    print(f"Model load failed: {e}")

try:
    processor = AutoProcessor.from_pretrained("facebook/sam3", trust_remote_code=True)
    print("Processor loaded successfully")
    print(type(processor))
except Exception as e:
    print(f"Processor load failed: {e}")
