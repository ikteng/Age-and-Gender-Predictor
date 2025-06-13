# build_age_model.py
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch
from torch import nn, optim
from tqdm import tqdm


IMAGE_DIR = 'crop_part1'
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs('models', exist_ok=True)
MODEL_PATH = f'models/age_resnet18_{IMG_SIZE}_{EPOCHS}.pth'

# Dataset
class AgeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label

def main():
    # Load image paths and age labels
    image_paths, ages = [], []
    for filename in os.listdir(IMAGE_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                age = int(filename.split('_')[0])
                image_paths.append(os.path.join(IMAGE_DIR, filename))
                ages.append(age)
            except:
                continue

    print(f"Loaded {len(image_paths)} images for age prediction.")
    print(f"Mean age: {np.mean(ages):.2f}, Std: {np.std(ages):.2f}")

    # # Show one sample image per age bucket
    # print("Displaying one sample image from each age bucket...")
    # age_bins = [(i, i + 9) for i in range(0, 100, 10)]
    # bucket_images = {}

    # # Collect only one image per bucket
    # for img, age in zip(images, ages):
    #     for low, high in age_bins:
    #         label = f"{low}-{high}"
    #         if label not in bucket_images and low <= age <= high:
    #             bucket_images[label] = img
    #             break

    # # Sort buckets by age range (ensure order)
    # sorted_buckets = sorted(bucket_images.items(), key=lambda x: int(x[0].split('-')[0]))

    # # Plotting
    # num_buckets = len(sorted_buckets)
    # fig, axes = plt.subplots(1, num_buckets, figsize=(num_buckets * 2, 3))

    # if num_buckets == 1:
    #     axes = [axes]  # ensure iterable if only 1 bucket

    # for ax, (bucket, img) in zip(axes, sorted_buckets):
    #     ax.imshow(img)
    #     ax.set_title(bucket)
    #     ax.axis("off")

    # plt.tight_layout()
    # plt.show()

    # # Age distribution in buckets of 10
    # print("Analyzing age distribution in buckets...")
    # bucket_counts = {f"{low}-{high}": 0 for low, high in age_bins}

    # for age in ages:
    #     for low, high in age_bins:
    #         if low <= age <= high:
    #             bucket_counts[f"{low}-{high}"] += 1
    #             break

    # total = len(ages)
    # print("Age distribution (by decade):")
    # for bucket, count in bucket_counts.items():
    #     print(f"  {bucket}: {count} ({count / total:.2%})")

    # Split
    print("Splitting into training and testing sets...")
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, ages, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(train_paths)}, Testing samples: {len(test_paths)}")


    print("Setting up data transformations...")
    # train_transform = transforms.Compose([
    #     transforms.Resize((IMG_SIZE, IMG_SIZE)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(15),
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    #     transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225]),
    # ])
    
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    print("Creating dataset and dataloader...")
    train_dataset = AgeDataset(train_paths, train_labels, transform=train_transform)
    test_dataset = AgeDataset(test_paths, test_labels, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count())
    print("Dataloaders ready.")

    # Model (ResNet18 backbone)
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)  # Single output for regression
    model = model.to(DEVICE)

    # Loss, optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        
        for inputs, labels in loop:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            
            # Update tqdm progress bar with loss
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")
    
    # Evaluation
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs).squeeze().cpu().numpy()
            preds.extend(outputs)
            targets.extend(labels.numpy())

    mae = np.mean(np.abs(np.array(preds) - np.array(targets)))
    print(f"Test MAE: {mae:.2f}")

    # Save model
    print("Saving model...")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()