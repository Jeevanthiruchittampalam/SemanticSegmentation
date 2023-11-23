import os
import shutil
import numpy as np

def divide_dataset(base_img_path, base_mask_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Ensure the ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1

    # List all files in the directories
    images = os.listdir(base_img_path)
    masks = os.listdir(base_mask_path)

    # Shuffle the files in a synchronized manner
    np.random.seed(42)  # For reproducible splits
    indices = np.arange(len(images))
    np.random.shuffle(indices)

    # Calculate split indices
    train_end = int(len(indices) * train_ratio)
    val_end = train_end + int(len(indices) * val_ratio)

    # Create subdirectories
    for split in ['train', 'val', 'test']:
        for folder in [base_img_path, base_mask_path]:
            os.makedirs(os.path.join(folder, split), exist_ok=True)

    # Distribute files
    for idx, (img, mask) in enumerate(zip(images, masks)):
        if idx < train_end:
            split = 'train'
        elif idx < val_end:
            split = 'val'
        else:
            split = 'test'

        # Copy images and masks to the respective directories
        shutil.copy(os.path.join(base_img_path, img), os.path.join(base_img_path, split, img))
        shutil.copy(os.path.join(base_mask_path, mask), os.path.join(base_mask_path, split, mask))

# Paths to your image and mask directories
base_img_path = 'dataset\semantic_drone_dataset\original_images'
base_mask_path = 'RGB_color_image_masks\RGB_color_image_masks'

divide_dataset(base_img_path, base_mask_path)
