from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            return image, path
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None, path

def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return [], []
    images, paths = zip(*batch)
    return list(images), list(paths)
