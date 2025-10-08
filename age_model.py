# train_age_model.py
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# resnet34: Training complete. Best model from epoch 14 with MAE: 4.05

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 30
PATIENCE = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = f"utkface_processed_{IMG_SIZE}.npz"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = f"age_resnet34_{IMG_SIZE}_{EPOCHS}.pth"

# --- Custom Dataset ---
class AgeDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # take an image and converts it to a PIL Image
        img = Image.fromarray(self.X[idx].astype('uint8')).convert("RGB")
        # applies transforms
        if self.transform:
            img = self.transform(img)
        # converts the label into a torch.tensor
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        return img, label

# --- Load preprocessed data ---
def load_data():
    print(f"Loading preprocessed data from {DATA_PATH} ...")
    data = np.load(DATA_PATH)
    X, y_age = data['X'], data['y_age']
    print(f"Loaded {X.shape[0]} samples.")
    return X, y_age

# --- Evaluation (MAE) ---
def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs).squeeze().cpu().numpy()
            preds.extend(outputs)
            targets.extend(labels.numpy())
    preds, targets = np.array(preds), np.array(targets)
    mae = np.mean(np.abs(preds - targets))
    return mae

# --- Train model ---
def train_model():
    X, y = load_data()

    # Split train/test
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Define transforms ---
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # --- Datasets & Loaders ---
    train_dataset = AgeDataset(train_X, train_y, transform=train_transform)
    test_dataset = AgeDataset(test_X, test_y, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Model ---
    # model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # model.fc = nn.Linear(model.fc.in_features, 1)
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    # model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    # model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model = model.to(DEVICE)

    # --- Loss & Optimizer ---
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_mae = float('inf')
    best_epoch = -1
    patience_counter = 0  # counter for early stopping

    print("Training started...\n")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        val_mae = evaluate(model, test_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_loss:.4f} | Val MAE: {val_mae:.2f}")

        # --- Save best model ---
        if val_mae < best_mae:
            best_mae = val_mae
            best_epoch = epoch + 1
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✅ Saved new best model at epoch {best_epoch} (MAE: {best_mae:.2f})")
            patience_counter = 0  # reset patience
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s).")

        # --- Early stopping ---
        if patience_counter >= PATIENCE:
            print(f"\n⏹️ Early stopping triggered at epoch {epoch+1}. Best MAE: {best_mae:.2f}")
            break

    print(f"\nTraining complete. Best model from epoch {best_epoch} with MAE: {best_mae:.2f}")
    print(f"Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
