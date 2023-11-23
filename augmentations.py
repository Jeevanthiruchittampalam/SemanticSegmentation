import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_training_augmentation():
    train_transform = [
        A.RandomCrop(width=256, height=256),  # Randomly crop the image
        A.HorizontalFlip(p=0.5),              # Flip the image horizontally with a 50% chance
        A.VerticalFlip(p=0.5),                # Flip the image vertically with a 50% chance
        A.Rotate(limit=35, p=0.5),            # Rotate the image up to 35 degrees with a 50% chance
        A.RandomBrightnessContrast(p=0.5),    # Apply random brightness and contrast adjustments
        A.CLAHE(p=0.5),                       # Apply Contrast Limited Adaptive Histogram Equalization
        A.GridDistortion(p=0.5),              # Apply grid distortion
        A.OpticalDistortion(p=0.5),           # Apply optical distortion
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # Normalize the image
        ToTensorV2()                          # Convert the image and mask to PyTorch tensors
    ]
    return A.Compose(train_transform)

def get_validation_augmentation():
    val_transform = [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # Normalize the image
        ToTensorV2()                          # Convert the image to a PyTorch tensor
    ]
    return A.Compose(val_transform)