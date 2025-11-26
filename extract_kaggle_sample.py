import zipfile
import random
import os
from pathlib import Path

def extract_random_images(zip_path, output_dir, num_images=10):
    """
    Extracts a specified number of random images from a zip file.
    """
    if not os.path.exists(zip_path):
        print(f"Error: Zip file not found at {zip_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Get list of all files in the zip
            all_files = z.namelist()
            
            # Filter for image files (assuming jpg, png, jpeg)
            image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('__MACOSX') and '/.' not in f]
            
            if not image_files:
                print("No image files found in the zip archive.")
                return

            print(f"Found {len(image_files)} images in the archive.")
            
            # Select random images
            selected_files = random.sample(image_files, min(num_images, len(image_files)))
            
            print(f"Extracting {len(selected_files)} images to {output_dir}...")
            
            for file_info in selected_files:
                # Extract the file
                z.extract(file_info, output_dir)
                print(f"Extracted: {file_info}")
                
    except zipfile.BadZipFile:
        print("Error: The file is not a valid zip file or is corrupted.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    zip_file_path = "/home/seunghyuk/workspace/study_ocr/data/clothing-dataset-full.zip"
    output_directory = "/home/seunghyuk/workspace/study_ocr/sample_images_from_kaggle"
    
    extract_random_images(zip_file_path, output_directory, num_images=10)
