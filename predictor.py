import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
from torch import nn


IMG_SIZE = 128
EPOCHS = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

AGE_MODEL_PATH = f"age_resnet34_{IMG_SIZE}_{EPOCHS}.pth"
GENDER_MODEL_PATH = f"gender_efficientnetb0_{IMG_SIZE}_{EPOCHS}.pth"

# --- Gender Mapping ---
genders = {0: "Male", 1: "Female"}

# --- Define Transform (same as test_transform used in training) ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# --- Load Age Model ---
def load_age_model():
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(AGE_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

# --- Load Gender Model ---
def load_gender_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model.load_state_dict(torch.load(GENDER_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

# --- Prediction from image ---
def predict_from_image(image_path, age_model, gender_model):
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Image not found or cannot be opened")
        return None, None

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    
    if len(faces) == 0:
        print("No face detected in the image.")
        return None, None

    # For simplicity, pick the largest detected face (like live webcam)
    x, y, w, h = max(faces, key=lambda b: b[2]*b[3])
    face_img = frame[y:y+h, x:x+w]
    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    img_tensor = transform(face_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # Gender prediction
        gender_logits = gender_model(img_tensor)
        gender_prob = torch.sigmoid(gender_logits).item()
        gender_label = genders[int(gender_prob > 0.5)]

        # Age prediction
        age_pred = age_model(img_tensor).item()

    # Optionally, draw rectangle like live webcam for visualization
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, f"{gender_label}, {int(age_pred)}", (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Show the image (optional)
    cv2.imshow("Predicted Face", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Predicted Gender: {gender_label} ({gender_prob:.2f})")
    print(f"Predicted Age: {age_pred:.1f} years")
    
    return age_pred, gender_label

# --- Live webcam prediction ---
def live_face_predict(age_model, gender_model):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            img_tensor = transform(face_pil).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                gender_logits = gender_model(img_tensor)
                gender_prob = torch.sigmoid(gender_logits).item()
                gender_label = genders[int(gender_prob > 0.5)]

                age_pred = age_model(img_tensor).item()

            text = f"{gender_label}, {int(age_pred)}"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Live Age & Gender Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load models
    print("Loading models...")
    age_model = load_age_model()
    gender_model = load_gender_model()

    # # Directory containing test images
    # test_dir = "test"
    # image_files = [f for f in os.listdir(test_dir) 
    #                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # # Predict each image
    # for img_file in image_files:
    #     img_path = os.path.join(test_dir, img_file)
    #     print(f"\nPredicting {img_file} ...")
    #     age_pred, gender_label = predict_from_image(img_path, age_model, gender_model)

    # Or run live webcam
    live_face_predict(age_model, gender_model)
