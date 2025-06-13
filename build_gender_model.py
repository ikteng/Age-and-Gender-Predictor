# build_gender_model.py
import os
import numpy as np
from collections import Counter
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm


IMAGE_DIR = 'crop_part1'
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs('models', exist_ok=True)
MODEL_PATH = f'models/gender_resnet18_{IMG_SIZE}_{EPOCHS}.pth'

class GenderDataset(Dataset):
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
        label = self.labels[idx]
        return img, label

def main():
    print("Loading images and extracting labels...")
    image_paths, genders = [], []
    for filename in os.listdir(IMAGE_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                gender = int(filename.split('_')[1])
                if gender in [0, 1]:
                    image_paths.append(os.path.join(IMAGE_DIR, filename))
                    genders.append(gender)
            except:
                continue

    print(f"Loaded {len(image_paths)} images.")

    # # Show sample male and female images
    # print("Displaying sample male and female images...")

    # # Convert to NumPy array for indexing
    # images = np.array(images)
    # genders = np.array(genders)

    # # Get indices for each gender
    # male_indices = np.where(genders == 0)[0]
    # female_indices = np.where(genders == 1)[0]

    # # Randomly select 5 from each
    # np.random.seed(42)
    # sample_male_idxs = np.random.choice(male_indices, size=5, replace=False)
    # sample_female_idxs = np.random.choice(female_indices, size=5, replace=False)

    # # Plotting
    # plt.figure(figsize=(12, 5))

    # # Male samples
    # for i, idx in enumerate(sample_male_idxs):
    #     plt.subplot(2, 5, i + 1)
    #     plt.imshow(images[idx])
    #     plt.title("Male")
    #     plt.axis("off")

    # # Female samples
    # for i, idx in enumerate(sample_female_idxs):
    #     plt.subplot(2, 5, i + 6)
    #     plt.imshow(images[idx])
    #     plt.title("Female")
    #     plt.axis("off")

    # plt.tight_layout()
    # plt.show()

    # Gender distribution
    gender_counts = Counter(genders)
    total = sum(gender_counts.values())
    print("Gender distribution:")
    for gender, count in gender_counts.items():
        label = "Male" if gender == 0 else "Female"
        print(f"  {label} ({gender}): {count} ({count / total:.2%})")

    print("Splitting into training and testing sets...")
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, genders, test_size=0.2, stratify=genders, random_state=42
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
    train_dataset = GenderDataset(train_paths, train_labels, transform=train_transform)
    test_dataset = GenderDataset(test_paths, test_labels, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count())
    print("Dataloaders ready.")

    print("Loading pretrained ResNet18 model...")
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Binary classification
    model = model.to(DEVICE)

    print("Computing class weights...")
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print("Setting up optimizer and scheduler...")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for batch_idx, (inputs, labels) in loop:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            # Update tqdm progress bar
            loop.set_postfix(loss=loss.item())

        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{EPOCHS} complete. Avg Loss: {epoch_loss:.4f}s")

    print("Training complete. Starting evaluation...")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Test accuracy: {acc:.4f}")

    print("Saving model...")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
