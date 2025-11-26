import os
import requests

# 이미지 저장 경로
save_dir = "/home/seunghyuk/workspace/study_ocr/sample_images"
os.makedirs(save_dir, exist_ok=True)

# 다운로드할 이미지 URL 리스트 (유사도 테스트를 위해 카테고리별로 구성)
image_urls = {
    "red_tshirt.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/24/Blue_Tshirt.jpg/800px-Blue_Tshirt.jpg", # Using blue as placeholder, will find better ones or stick to generic
    "blue_jeans.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Jeans_for_men.jpg/800px-Jeans_for_men.jpg",
    "white_dress.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/Wedding_dress_displayed_at_shop.jpg/449px-Wedding_dress_displayed_at_shop.jpg",
    "leather_jacket.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/Leather_jacket_01.jpg/800px-Leather_jacket_01.jpg",
    "sneakers.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/73/Pair_of_sneakers.jpg/800px-Pair_of_sneakers.jpg",
    "hoodie.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/Gray_hoodie.jpg/800px-Gray_hoodie.jpg",
    "skirt.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Skirt.jpg/800px-Skirt.jpg",
    "suit.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/16/Man_in_suit_2.jpg/450px-Man_in_suit_2.jpg",
    "coat.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/Coat_in_shop.jpg/800px-Coat_in_shop.jpg",
    "floral_dress.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Floral_Dress.jpg/800px-Floral_Dress.jpg"
}

# Correcting the first one to be actual red tshirt if possible, otherwise generic tshirt
image_urls["red_tshirt.jpg"] = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/Red_T-shirt.jpg/800px-Red_T-shirt.jpg"


print(f"Downloading {len(image_urls)} images to {save_dir}...")

for filename, url in image_urls.items():
    file_path = os.path.join(save_dir, filename)
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {filename}")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")

print("Done.")
