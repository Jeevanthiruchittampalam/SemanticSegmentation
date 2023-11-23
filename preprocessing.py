import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from augmentations import get_training_augmentation, get_validation_augmentation

class_dict = pd.read_csv('class_dict_seg.csv')


def rgb_to_label(mask_path, class_dict):
    #print("Entering rgb_to_label function")
    mask = Image.open(mask_path)
    mask = np.array(mask)

    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=int)

    for index, row in class_dict.iterrows():
        #print(row)  # This will print each row being accessed
        # Accessing color values using iloc
        r, g, b = row.iloc[1], row.iloc[2], row.iloc[3]
        class_id = index
        label_mask[np.where((mask == [r, g, b]).all(axis=2))] = class_id

    return label_mask




def resize_image(image_path, size):
    image = Image.open(image_path)
    image = image.resize((size, size), Image.ANTIALIAS)
    return image

class SemanticDroneDataset(Dataset):
    def __init__(self, base_img_dir, base_mask_dir, class_dict):
        self.base_img_dir = base_img_dir
        self.base_mask_dir = base_mask_dir
        self.class_dict = class_dict
        self.images = os.listdir(self.base_img_dir)
        self.augmentation = get_training_augmentation() if "train" in self.base_img_dir else get_validation_augmentation()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_img_dir, self.images[idx])
        mask_path = os.path.join(self.base_mask_dir, self.images[idx].replace('.jpg', '.png'))
        image = np.array(Image.open(img_path).convert("RGB"))

        #print(f"Processing image: {img_path}")  # Add this line
        mask = rgb_to_label(mask_path, self.class_dict)
        unique_labels = np.unique(mask)
        print(f"Unique labels in the mask: {unique_labels}")

        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask



transform = transforms.Compose([
    transforms.ToTensor(),
    # Add any other transformations here
])

# Remove the transform parameter
dataset = SemanticDroneDataset(base_img_dir='dataset/semantic_drone_dataset/original_images', 
                               base_mask_dir='RGB_color_image_masks/RGB_color_image_masks', 
                               class_dict=class_dict)

data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

