# evaluate_model.py
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score

# --- Configuration ---
IMG_SIZE = 128
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = f"utkface_processed_{IMG_SIZE}.npz"

# --- Dataset Classes ---
class AgeDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = Image.fromarray(self.X[idx].astype("uint8")).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        return img, label


class GenderDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = Image.fromarray(self.X[idx].astype("uint8")).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        return img, label


# --- Load data ---
def load_data():
    print(f"Loading data from {DATA_PATH} ...")
    data = np.load(DATA_PATH)
    X, y_age, y_gender = data["X"], data["y_age"], data["y_gender"]
    print(f"Loaded {X.shape[0]} total samples.")
    return X, y_age, y_gender


# --- Transforms ---
def get_test_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# --- Evaluate Age Model ---
def evaluate_age(model_path):
    X, y_age, _ = load_data()
    _, test_X, _, test_y = train_test_split(X, y_age, test_size=0.2, random_state=42)

    test_dataset = AgeDataset(test_X, test_y, transform=get_test_transform())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = models.resnet34(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    preds, targets = [], []
    print(f"Evaluating age model ({model_path}) ...")
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating"):
            imgs = imgs.to(DEVICE)
            outputs = model(imgs).squeeze().cpu().numpy()
            preds.extend(outputs)
            targets.extend(labels.numpy())

    mae = mean_absolute_error(targets, preds)
    print(f"\n📊 Age Model Evaluation:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    return mae


# --- Evaluate Gender Model ---
def evaluate_gender(model_path):
    X, _, y_gender = load_data()
    _, test_X, _, test_y = train_test_split(X, y_gender, test_size=0.2, random_state=42, stratify=y_gender)

    test_dataset = GenderDataset(test_X, test_y, transform=get_test_transform())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    preds, targets = [], []
    print(f"Evaluating gender model ({model_path}) ...")
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating"):
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds.extend((probs > 0.5).astype(int))
            targets.extend(labels.numpy().astype(int))

    acc = accuracy_score(targets, preds)
    print(f"\n📊 Gender Model Evaluation:")
    print(f"Accuracy: {acc * 100:.2f}%")
    return acc


# --- Main ---
if __name__ == "__main__":
    print("Select model type to evaluate:")
    print("1. Age Model")
    print("2. Gender Model")
    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        model_path = f"age_resnet34_{IMG_SIZE}_30.pth"
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
        else:
            evaluate_age(model_path)

    elif choice == "2":
        model_path = f"gender_efficientnetb0_{IMG_SIZE}_30.pth"
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
        else:
            evaluate_gender(model_path)

    else:
        print("Invalid choice.")
