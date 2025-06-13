# predictor.py
import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, models
from torch import nn
import mediapipe as mp


IMG_SIZE = 128
TEST_FOLDER = 'test'
EPOCHS = 20
AGE_MODEL_PATH = f'models/age_resnet18_{IMG_SIZE}_{EPOCHS}.pth'
GENDER_MODEL_PATH = f'models/gender_resnet18_{IMG_SIZE}_{EPOCHS}.pth'
GENDER_MAP = {0: "Male", 1: "Female"}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load CNN-based age model
age_model = models.resnet18(pretrained=False)
age_model.fc = nn.Linear(age_model.fc.in_features, 1)
age_model.load_state_dict(torch.load(AGE_MODEL_PATH, map_location=DEVICE))
age_model.to(DEVICE)
age_model.eval()

# Load gender model (PyTorch)
gender_model = models.resnet18(pretrained=False)
gender_model.fc = nn.Linear(gender_model.fc.in_features, 2)
gender_model.load_state_dict(torch.load(GENDER_MODEL_PATH, map_location=DEVICE))
gender_model.to(DEVICE)
gender_model.eval()

# Define transform for gender model
common_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

# Function to predict age and gender
def process_and_predict(face_image):
    """Resize, normalize, and predict age and gender."""
    # === Age Prediction ===
    age_img = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
    age_tensor = common_transform(age_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        age_pred = age_model(age_tensor).cpu().item()
    age = int(round(age_pred))

    # === Gender Prediction ===
    gender_img = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
    input_tensor = common_transform(gender_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = gender_model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
        gender_class = np.argmax(probs)

    gender_label = GENDER_MAP[gender_class]
    return int(round(age)), gender_label, probs

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Main loop
with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    for filename in os.listdir(TEST_FOLDER):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        path = os.path.join(TEST_FOLDER, filename)
        image = cv2.imread(path)
        if image is None:
            print(f"Could not read image: {filename}")
            continue

        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)

                # Optional padding around face
                padding = int(0.1 * w)
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(iw, x + w + padding)
                y2 = min(ih, y + h + padding)

                face = image[y1:y2, x1:x2]

                # Only predict if face crop is valid
                if face.size == 0:
                    continue

                age, gender, probs = process_and_predict(face)

                label = f"Age: {age}, Gender: {gender}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

        else:
            print(f"No face detected in {filename}")

        # Show result
        cv2.imshow(f'Prediction - {filename}', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
