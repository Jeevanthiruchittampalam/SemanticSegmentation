import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocessing import SemanticDroneDataset, class_dict
from UNET import UNet_2D  # Ensure this points to your U-Net model


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("cuda")

# Hyperparameters
num_classes = 24  # Update based on your number of classes
learning_rate = 1e-3
batch_size = 4
num_epochs = 25

# Paths to your datasets
train_img_dir = 'dataset/semantic_drone_dataset/original_images/train'
train_mask_dir = 'RGB_color_image_masks/RGB_color_image_masks/train'
val_img_dir = 'dataset/semantic_drone_dataset/original_images/val'
val_mask_dir = 'RGB_color_image_masks/RGB_color_image_masks/val'

# Load Data
train_dataset = SemanticDroneDataset(base_img_dir=train_img_dir, base_mask_dir=train_mask_dir, class_dict=class_dict)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = SemanticDroneDataset(base_img_dir=val_img_dir, base_mask_dir=val_mask_dir, class_dict=class_dict)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)


# Initialize network
model = UNet_2D(num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  # Or another suitable loss function for your task
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    # Training step
    model.train()
    train_loss = 0
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device, dtype=torch.long)  # Ensure targets are long tensors

        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        train_loss += loss.item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation step
    model.eval()
    val_loss = 0
    for batch_idx, (data, targets) in enumerate(val_loader):
        data = data.to(device)
        targets = targets.to(device, dtype=torch.long)

        scores = model(data)
        loss = criterion(scores, targets)
        val_loss += loss.item()

        # (Optional) Add accuracy calculation code here

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}")

# Save Model
torch.save(model.state_dict(), 'unet_model.pth')
