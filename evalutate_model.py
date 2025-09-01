# evaluate_model.py
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch import nn
from tqdm import tqdm
from build_age_model import AgeDataset
from build_gender_model import GenderDataset

# ===== CONFIG =====
IMAGE_DIR = 'crop_part1'
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

AGE_MODEL_PATH = f'models/age_resnet18_{IMG_SIZE}_{EPOCHS}.pth'
GENDER_MODEL_PATH = f'models/gender_resnet18_{IMG_SIZE}_{EPOCHS}.pth'

# ===== COMMON FUNCTIONS =====
def load_age_data():
    paths, labels = [], []
    for f in os.listdir(IMAGE_DIR):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                paths.append(os.path.join(IMAGE_DIR, f))
                labels.append(int(f.split('_')[0]))
            except:
                continue
    return paths, labels

def load_gender_data():
    paths, labels = [], []
    for f in os.listdir(IMAGE_DIR):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                gender = int(f.split('_')[1])
                if gender in [0, 1]:
                    paths.append(os.path.join(IMAGE_DIR, f))
                    labels.append(gender)
            except:
                continue
    return paths, labels

def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

# ===== EVALUATE AGE MODEL =====
def evaluate_age_model():
    print("\n🔹 Evaluating AGE model...")
    paths, ages = load_age_data()
    _, test_paths, _, test_labels = train_test_split(paths, ages, test_size=0.2, random_state=42)

    dataset = AgeDataset(test_paths, test_labels, transform=get_transform())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(AGE_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

    preds, targets = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Predicting Ages"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs).squeeze().cpu().numpy()
            preds.extend(outputs)
            targets.extend(labels.numpy())

    preds, targets = np.array(preds), np.array(targets)
    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2 = r2_score(targets, preds)

    print("\nAGE MODEL RESULTS")
    print(f"MAE  : {mae:.2f}") # Mean Absolute Error: average absolute difference between predicted ages and actual ages
    print(f"RMSE : {rmse:.2f}") # Root Mean Squared Error: square root of the average of squared errors
    print(f"R²   : {r2:.4f}") # Coefficient of Determination: Measures how well the model explains the variance in the data

# ===== EVALUATE GENDER MODEL =====
def evaluate_gender_model():
    print("\n🔹 Evaluating GENDER model...")
    paths, genders = load_gender_data()
    _, test_paths, _, test_labels = train_test_split(paths, genders, test_size=0.2, stratify=genders, random_state=42)

    dataset = GenderDataset(test_paths, test_labels, transform=get_transform())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(GENDER_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

    preds, targets = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Predicting Genders"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            targets.extend(labels.numpy())

    acc = accuracy_score(targets, preds)
    print("\nGENDER MODEL RESULTS")
    print(f"Accuracy : {acc:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(targets, preds, target_names=["Male", "Female"]))
    # Precision: Out of all predicted positives, how many were actually correct?
    # Recall (Sensitivity or True Positive Rate): Out of all actual positives, how many did the model correctly identify?
    # F1-score: Harmonic mean of precision and recall (balances both)
    # Accuracy:  Overall proportion of correct predictions

# ===== MAIN =====
if __name__ == "__main__":
    if os.path.exists(AGE_MODEL_PATH):
        evaluate_age_model()
    else:
        print("Age model not found, skipping age evaluation.")

    if os.path.exists(GENDER_MODEL_PATH):
        evaluate_gender_model()
    else:
        print("Gender model not found, skipping gender evaluation.")
