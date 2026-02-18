import os
import shutil
import random
from tqdm import tqdm

def get_image_files(directory):
    """
    Recursively finds all image files in a directory.
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG')
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(valid_extensions):
                image_files.append(os.path.join(root, file))
    return image_files

def split_data(source_dir, target_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Splits data from source_dir into target_dir/train, target_dir/val, target_dir/test.
    """
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    os.makedirs(target_dir, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        for cls in classes:
            os.makedirs(os.path.join(target_dir, split, cls), exist_ok=True)
            
    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        images = get_image_files(cls_path)
        random.shuffle(images)
        
        num_images = len(images)
        if num_images == 0:
            print(f"Warning: No images found for class {cls}")
            continue
            
        train_end = int(num_images * train_ratio)
        val_end = train_end + int(num_images * val_ratio)
        
        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]
        
        splits = {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }
        
        print(f"Splitting class {cls}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
        
        for split_name, split_imgs in splits.items():
            for i, src_path in enumerate(tqdm(split_imgs, desc=f"Copying {split_name} images for {cls}")):
                # Use a unique name to avoid collisions if files had same name in different subdirs
                filename = f"{cls}_{split_name}_{i}_{os.path.basename(src_path)}"
                dst_path = os.path.join(target_dir, split_name, cls, filename)
                shutil.copy(src_path, dst_path)

if __name__ == "__main__":
    SOURCE = "data/raw"
    TARGET = "data/processed"
    # Ensure target is clean if we are re-running
    if os.path.exists(TARGET):
        print(f"Cleaning existing target directory: {TARGET}")
        shutil.rmtree(TARGET)
    
    split_data(SOURCE, TARGET)
