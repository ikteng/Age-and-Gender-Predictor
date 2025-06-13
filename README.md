# Age and Gender Predictor


![UTKface dataset](https://github.com/ikteng/Age-and-Gender-Predictor/blob/aea2f51ddd94cc3d3abf2e661136da0e00d76730/media/Screenshot%202024-05-21%20113638.png)

Dataset: https://www.kaggle.com/datasets/jangedoo/utkface-new

I used the crop_part1 of the dataset in this project.

# How to Use This Repository?
1. clone the repository
    ```bash
    git clone https://github.com/ikteng/Age-and-Gender-Predictor.git
    cd Age-and-Gender-Predictor
    ```
2. Install Dependencies
    All necessary Python packages are listed in the requirements.txt file. To install them: `pip install -r requirements.txt`
3. Download the Dataset
    This project uses the UTKFace dataset (https://www.kaggle.com/datasets/jangedoo/utkface-new). After downloading, extract and place the crop part1 folder in the root of the project directory
4. Run the program
    You can run the build_model files to run and build models or run the prediction/prediction live files to test the models out yourself!

## Building the Model

### Age Model
The age prediction model is built using a ResNet18-based convolutional neural network trained on the UTKFace dataset, specifically the crop_part1 subset. This model treats age prediction as a regression problem, where the goal is to predict a continuous age value from a face image.

#### Overview
1. Data Preparation
    Each image filename in the dataset encodes the age label as the first component (e.g., 25_1_0_20170116174525125.jpg corresponds to age 25). The script extracts age values and associates them with their respective image paths.

2. Dataset and Dataloader
    A custom AgeDataset class is implemented to handle image loading and transformation. The dataset is split into training (80%) and testing (20%) subsets using train_test_split.

3. Image Preprocessing
    Images are resized to 128×128 pixels. The training set undergoes basic data augmentation such as horizontal flipping, while both training and testing images are normalized using ImageNet mean and standard deviation values.

4. Model Architecture
     model is based on a pre-trained ResNet18 architecture from torchvision.models. The final fully connected layer is modified to output a single value (i.e., the predicted age) instead of a classification.

5. Training
    For each epoch, the model is trained to minimize Mean Squared Error (MSE) loss. Loss statistics are tracked using tqdm for a real-time progress bar.

6. Evaluation
    After training, the model is evaluated on the test set by calculating the Mean Absolute Error (MAE) between predicted and actual age values.

7. Model Saving
    The trained model is saved as a .pth file under the models/ directory

### Gender Model
The gender prediction model uses a transfer learning approach, leveraging a pre-trained ResNet18 convolutional neural network. The model classifies a face as either male or female based on facial features extracted from images in the UTKFace dataset.

#### Overview
1. Label Extraction
    Each image filename in the dataset includes encoded metadata, where the second component indicates gender (0 for male, 1 for female). The script reads image filenames from the crop_part1 folder and extracts gender labels accordingly.

2. Dataset Construction
A custom GenderDataset class is used to load images and their corresponding gender labels. The dataset is split into training and testing sets using stratified sampling (80/20 split), ensuring balanced class distribution.

3. Image Preprocessing
    Images are resized to 128×128 pixels. The training images are augmented with random horizontal flips to improve generalization. All images are normalized using the ImageNet mean and standard deviation.

4. Model Architecture
    A ResNet18 model from torchvision.models is used with pre-trained weights. The final fully connected layer (fc) is replaced with a new layer outputting two values corresponding to the two gender classes.The model is trained using weighted cross-entropy loss to handle potential class imbalance.

5. Training
    The model is trained for 20 epochs with the Adam optimizer and a learning rate scheduler (reducing LR every 3 epochs).
    Training progress, including the loss for each batch, is tracked with tqdm progress bars.
    Class weights are automatically computed to improve learning on imbalanced data.

6. Evaluation
    After training, the model is evaluated on the test set. Performance is measured using accuracy score between predicted and ground truth labels.

7. Model Saving
    The trained model is saved in the models/ directory


## Making Predictions

### Predictions from Facial Images (predictor.py)
The predictor.py script performs offline batch prediction of both age and gender from face images stored in a local directory (test/). It uses two pre-trained ResNet18 models: one for regressing age and one for classifying gender, along with MediaPipe for face detection.

#### Workflow
1. Model Loading
    Both age and gender models are loaded from the models/ directory. Each model is based on a ResNet18 backbone with modified output layers (1 output for age, 2 outputs for gender classes).Models are moved to GPU if available (cuda), else fallback to CPU.

2. Face Detection
    - Uses MediaPipe’s FaceDetection with a confidence threshold of 0.5.
    - For each detected face, the bounding box is converted from relative coordinates to absolute image space.
    - A small padding is applied around each bounding box to better crop the full facial region.
    - Extracts bounding boxes for detected faces and crops the regions of interest (ROI).

3. Image Preprocessing
    The cropped face is:
    - Converted from BGR (OpenCV) to RGB (PIL).
    - Resized and center-cropped to 128×128 pixels.
    - Normalized using ImageNet’s mean and standard deviation.
    - The transformed image is passed into both models.

4. Prediction Logic
    - Age Prediction: The face tensor is passed into the regression model, and the output is rounded to the nearest integer.
    - Gender Prediction: The face tensor is passed into the classification model, followed by a softmax layer to obtain probabilities for each class.

5. Post-processing
    Predicted labels are overlaid on the original image using OpenCV:
    - Age: Integer value.
    - Gender: "Male" or "Female", based on maximum softmax score.
    A bounding box and label are drawn around the detected face.

#### Usage
1. Place test images in the test/ directory.
2. Run predictor.py; the script processes each image, displays the annotated result, and prints warnings if no face is detected.

#### Example Output
![prediction](https://github.com/ikteng/Age-and-Gender-Predictor/blob/aea2f51ddd94cc3d3abf2e661136da0e00d76730/media/Screenshot%202024-05-21%20114127.png)

For each test image:
    - Faces are detected and annotated with bounding boxes.
    - Labels such as Age: 26, Gender: Female are shown above each face.
    - Results are visualized one at a time using cv2.imshow().

### Predictions from Live Camera (predictor live.py)
The predictor_live.py script performs real-time age and gender prediction from a webcam feed using trained ResNet18 models and MediaPipe for face detection. It provides an intuitive, live demo where bounding boxes and predicted labels are displayed over detected faces.

#### Workflow
1. Model Loading
    Both age and gender models are loaded from the models/ directory. Each model is based on a ResNet18 backbone with modified output layers (1 output for age, 2 outputs for gender classes).Models are moved to GPU if available (cuda), else fallback to CPU.

2. Face Detection
    - Uses MediaPipe’s FaceDetection with a confidence threshold of 0.5.
    - For each detected face, the bounding box is converted from relative coordinates to absolute image space.
    - A small padding is applied around each bounding box to better crop the full facial region.

3. Image Preprocessing
    The cropped face is:
    - Converted from BGR (OpenCV) to RGB (PIL).
    - Resized and center-cropped to 128×128 pixels.
    - Normalized using ImageNet’s mean and standard deviation.
    - The transformed image is passed into both models.

4. Prediction Logic
    - Age: Output from the age model is treated as a single scalar and rounded to the nearest integer.
    - Gender: A softmax is applied to the gender logits to produce class probabilities. The class with the highest probability is selected, and both percentages (male/female) are displayed.

5. Output Display
    - The webcam stream is displayed live.
    - For every detected face, a green bounding box is drawn, and a label.
    - The prediction updates in real time as the video progresses.