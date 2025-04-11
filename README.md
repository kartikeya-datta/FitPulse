# Exercise Position Detection

# Installation Guide

## Step 1: Download the Repository
- Save the repository to your desired working directory on your system.

## Step 2: Install Anaconda
- If you haven’t already, download and install Anaconda from the official website.

## Step 3: Open Anaconda Prompt
- Launch the Anaconda Prompt and navigate to your working directory using the `cd` command.

## Step 4: Set Up the Conda Environment
- Run the following command in the Anaconda Prompt to create a virtual environment:
  ```bash
  conda env create --name <ENV_NAME> --file environment.yml

## Step 5: Activate the Conda Environment
- Use the following command to activate the newly created Conda environment:
  ```bash
  conda activate <ENV_NAME>

## Step 6: Go to Anaconda Navigator and select the env name you created

## Step 7: Navigate to your convenient bash and run the command
- Use the following command to start your app:
  ```bash
   python app.py


===============================================================

** Step-by-Step Breakdown of the Exercise Detection Jupyter Notebook **

0. Import Dependencies
This section imports all the necessary Python libraries and modules:

Computer Vision: cv2 (OpenCV) for image processing and webcam access, mediapipe for pose estimation.
Numerical Computing: numpy for array operations, math for calculations.
Visualization: matplotlib.pyplot for plotting (though not heavily used here).
File Handling: os for file and directory operations.
Machine Learning:
tensorflow (and its submodules like keras) for building and training neural networks.
sklearn for data splitting (train_test_split), evaluation metrics (accuracy_score, classification_report), and confusion matrices (multilabel_confusion_matrix).
Others: time for timing, scipy.stats (not heavily used), and warning suppression utilities.
Key actions:

Suppresses TensorFlow warnings and sets up the environment for cleaner output.
1. Keypoints using MP Pose
This section sets up MediaPipe's Pose estimation model to detect human keypoints (e.g., shoulders, elbows) from video frames.

mp_pose and mp_drawing: Initializes MediaPipe's pre-trained pose model and drawing utilities.
mediapipe_detection: A function that:
Converts the image from BGR (OpenCV format) to RGB (MediaPipe format).
Processes the image with the pose model to detect keypoints.
Returns the processed image and results (keypoints).
draw_landmarks: Draws detected keypoints and connections (e.g., lines between joints) on the image using custom colors and styles.
Webcam Test: Opens the webcam, captures frames, detects poses, draws landmarks, and displays the result in a window. Press 'q' to exit.
2. Extract Keypoints
This section processes the keypoints detected by MediaPipe for use in machine learning.

Keypoint Collection: Loops through the results.pose_landmarks.landmark to extract (x, y, z, visibility) for each of the 33 keypoints.
num_landmarks, num_values, num_input_values: Calculates the total input size (33 landmarks × 4 values = 132).
extract_keypoints: A function that flattens the keypoints into a 1D array (132 elements) or returns zeros if no landmarks are detected. This is the input format for the neural network.
3. Setup Folders for Collection
This section prepares a directory structure to store training data as NumPy arrays.

DATA_PATH: Sets the directory to store data (e.g., data/ in the current working directory).
actions: Defines the exercises to detect: curl, press, squat.
no_sequences: Number of video sequences (50) per action.
sequence_length: Frames per video (e.g., 30 frames at 30 FPS = 1 second).
Folder Creation: Creates subdirectories like data/curl/101/, data/press/101/, etc., for storing keypoints frame-by-frame.
4. Collect Keypoint Values for Training
This section captures webcam data for training.

colors: Assigns colors to each exercise for visualization (e.g., orange for curls).
Data Collection Loop:
Loops through each action (curl, press, squat).
For each action, records 50 sequences (videos).
For each sequence, captures sequence_length frames.
Detects keypoints, draws landmarks, and saves keypoints as .npy files in the folder structure (e.g., data/curl/101/0.npy).
Visualization: Displays text like "Collecting curl Video #101" on the screen with a delay at the start of each sequence.
5. Preprocess Data and Create Labels/Features
This section loads and organizes the collected data for training.

label_map: Maps actions to integers (e.g., curl: 0, press: 1, squat: 2).
Data Loading: Loops through the saved .npy files, builds sequences (list of frames), and assigns labels.
X and y:
X is a 3D array (num_sequences, sequence_length, 132) of keypoints.
y is a one-hot encoded array (num_sequences, num_classes) of labels (e.g., [1, 0, 0] for curl).
Train/Test Split: Splits data into training (90%) and testing (10%), then further splits training into training (75%) and validation (15%).
6. Build and Train Neural Networks
This section defines and trains two models: a basic LSTM and an LSTM with attention.

6a. LSTM
Callbacks: Early stopping, learning rate reduction, model checkpointing, and TensorBoard logging.
Model Architecture:
3 LSTM layers (128, 256, 128 units) with relu activation.
3 Dense layers (128, 64, 3 units) with softmax for classification.
Training: Compiles with Adam optimizer and categorical crossentropy loss, trains for up to 500 epochs with callbacks.
6b. LSTM + Attention
Attention Mechanism: Implements Luong’s attention to focus on important frames.
Model Architecture:
Bidirectional LSTM (256 hidden units).
Attention layer.
Dense layers (512 units with dropout, 3 units with softmax).
Training: Similar to the LSTM, but interrupted by a KeyboardInterrupt in the output.
7. Save and Load Weights
Save: Saves each model's weights as .h5 files (e.g., LSTM.h5).
Load: Loads weights back into the models (requires rebuilding the model first).
8. Make Predictions
Predicts on the test set (X_test) using both models and stores results.
9. Evaluations
Evaluates model performance using confusion matrices, accuracy, precision, recall, and F1-score.

Confusion Matrices: Shows true vs. predicted labels for each class.
Accuracy: Percentage of correct predictions (e.g., LSTM: 86.67%, Attention: 100%).
Precision/Recall/F1: Weighted averages across classes (Attention model scores perfectly due to small test set).
10. Choose Model
Selects the LSTM model for real-time testing (could be changed to Attention).

11. Calculate Joint Angles & Count Reps
Adds logic to track exercise repetitions based on joint angles.

calculate_angle: Computes the angle between three 3D points (e.g., shoulder-elbow-wrist).
get_coordinates: Extracts (x, y) coordinates for specific joints (e.g., left elbow).
viz_joint_angle: Displays the angle on the image near the joint.
count_reps: Tracks reps for each exercise:
Curl: Counts when elbow angle goes from <30° (up) to >140° (down).
Press: Uses elbow angle and shoulder-elbow/wrist distances.
Squat: Monitors knee and hip angles bilaterally.
12. Real-Time Testing
Runs the full system in real-time using the webcam.

prob_viz: Displays a probability bar graph for each action.
Main Loop:
Captures frames, detects poses, extracts keypoints.
Predicts the current action using the model when sequence_length frames are collected.
Updates rep counters and visualizes probabilities, angles, and counts on the screen.
Saves the output as a video file (e.g., lstm_real_time_test.avi).
Exit: Press 'q' to stop.
