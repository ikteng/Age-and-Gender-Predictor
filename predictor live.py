# predictor_live.py
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, models
from torch import nn
import mediapipe as mp

# ==== Config ====
IMG_SIZE = 128
EPOCHS = 20
AGE_MODEL_PATH = f'models/age_resnet18_{IMG_SIZE}_{EPOCHS}.pth'
GENDER_MODEL_PATH = f'models/gender_resnet18_{IMG_SIZE}_{EPOCHS}.pth'
GENDER_MAP = {0: "Male", 1: "Female"}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==== Load Models ====
# Age model
age_model = models.resnet18(pretrained=False)
age_model.fc = nn.Linear(age_model.fc.in_features, 1)
age_model.load_state_dict(torch.load(AGE_MODEL_PATH, map_location=DEVICE))
age_model.to(DEVICE)
age_model.eval()

# Gender model
gender_model = models.resnet18(pretrained=False)
gender_model.fc = nn.Linear(gender_model.fc.in_features, 2)
gender_model.load_state_dict(torch.load(GENDER_MODEL_PATH, map_location=DEVICE))
gender_model.to(DEVICE)
gender_model.eval()

# ==== Image Transform ====
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ==== Prediction Function ====
def predict_from_face(face_image):
    image_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
    input_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        age = age_model(input_tensor).cpu().item()
        gender_logits = gender_model(input_tensor)
        gender_probs = torch.nn.functional.softmax(gender_logits, dim=1).cpu().numpy()[0]
        gender = GENDER_MAP[np.argmax(gender_probs)]

    return int(round(age)), gender, gender_probs

# ==== Initialize MediaPipe ====
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# ==== Webcam Stream ====
cap = cv2.VideoCapture(0)  # Use 0 for default webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = int(bbox.xmin * iw)
                y = int(bbox.ymin * ih)
                w = int(bbox.width * iw)
                h = int(bbox.height * ih)

                padding = int(0.1 * w)
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(iw, x + w + padding)
                y2 = min(ih, y + h + padding)

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                age, gender, gender_probs = predict_from_face(face)
                male_pct = gender_probs[0] * 100
                female_pct = gender_probs[1] * 100
                label = f"Age: {age}, Gender: {gender} (M: {male_pct:.1f}%, F: {female_pct:.1f}%)"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('Live Age & Gender Prediction', frame)

        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:  # ESC or 'q' to exit
            break

cap.release()
cv2.destroyAllWindows()
