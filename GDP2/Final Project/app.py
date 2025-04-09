from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
import numpy as np
import math
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Flatten, Bidirectional, Permute, multiply
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

## Build and Load Model
def attention_block(inputs, time_steps):
    """
    Attention layer for deep neural network
    """
    # Attention weights
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    
    # Attention vector
    a_probs = Permute((2, 1), name='attention_vec')(a)
    
    # Luong's multiplicative score
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul') 
    
    return output_attention_mul

def build_model(HIDDEN_UNITS=256, sequence_length=30, num_input_values=33*4, num_classes=3):
    """
    Function used to build the deep neural network model on startup
    """
    inputs = Input(shape=(sequence_length, num_input_values))
    # Bi-LSTM
    lstm_out = Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True))(inputs)
    # Attention
    attention_mul = attention_block(lstm_out, sequence_length)
    attention_mul = Flatten()(attention_mul)
    # Fully Connected Layer
    x = Dense(2*HIDDEN_UNITS, activation='relu')(attention_mul)
    x = Dropout(0.5)(x)
    # Output
    x = Dense(num_classes, activation='softmax')(x)
    # Bring it all together
    model = Model(inputs=[inputs], outputs=x)

    ## Load Model Weights
    load_dir = "./models/LSTM_II.h5"  
    try:
        model.load_weights(load_dir)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None
    
    return model

# Initialize model
model = build_model()

if model is None:
    logging.error("Model failed to load. Exiting application.")
    exit()

## Mediapipe Initialization
mp_pose = mp.solutions.pose  # Pre-trained pose estimation model from Google Mediapipe
mp_drawing = mp.solutions.drawing_utils  # Supported Mediapipe visualization tools
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)  # Initial thresholds

## Video Capture
cap = cv2.VideoCapture(0)  # 0 for default webcam

# Define global variables for settings
threshold1 = 0.5  # Minimum Keypoint Detection Confidence
threshold2 = 0.5  # Minimum Tracking Confidence
threshold3 = 0.5  # Minimum Activity Classification Confidence

## Real Time Machine Learning and Computer Vision Processes
class VideoProcessor:
    def __init__(self, threshold1, threshold2, threshold3):
        # Update pose model with new thresholds
        self.pose = mp_pose.Pose(min_detection_confidence=threshold1, min_tracking_confidence=threshold2)
        
        # Parameters
        self.actions = np.array(['curl', 'press', 'squat'])
        self.sequence_length = 30
        self.colors = [(245,117,16), (117,245,16), (16,117,245)]
        self.threshold = threshold3
        
        # Detection variables
        self.sequence = []
        self.current_action = ''

        # Rep counter logic variables
        self.curl_counter = 0
        self.press_counter = 0
        self.squat_counter = 0
        self.curl_stage = None
        self.press_stage = None
        self.squat_stage = None

    def draw_landmarks(self, image, results):
        """
        Draws keypoints and landmarks detected by the human pose estimation model
        """
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
        return image

    def extract_keypoints(self, results):
        """
        Processes and organizes the keypoints detected from the pose estimation model 
        to be used as inputs for the exercise decoder models
        """
        if results.pose_landmarks:
            keypoints = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
        else:
            keypoints = np.zeros(33*4)
        return keypoints

    def calculate_angle(self, a, b, c):
        """
        Computes 3D joint angle inferred by 3 keypoints and their relative positions to one another
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        
        return angle 

    def get_coordinates(self, landmarks, mp_pose, side, joint):
        """
        Retrieves x and y coordinates of a particular keypoint from the pose estimation model
        """
        coord = getattr(mp_pose.PoseLandmark, f"{side.upper()}_{joint.upper()}")
        x_coord_val = landmarks[coord.value].x
        y_coord_val = landmarks[coord.value].y
        return [x_coord_val, y_coord_val] 

    def viz_joint_angle(self, image, angle, joint):
        """
        Displays the joint angle value near the joint within the image frame
        """
        cv2.putText(image, str(int(angle)), 
                    tuple(np.multiply(joint, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        return image

    def count_reps(self, image, landmarks, mp_pose):
        """
        Counts repetitions of each exercise. Global count and stage (i.e., state) variables are updated within this function.
        """
        if self.current_action == 'curl':
            # Get coords
            shoulder = self.get_coordinates(landmarks, mp_pose, 'left', 'shoulder')
            elbow = self.get_coordinates(landmarks, mp_pose, 'left', 'elbow')
            wrist = self.get_coordinates(landmarks, mp_pose, 'left', 'wrist')
            
            # calculate elbow angle
            angle = self.calculate_angle(shoulder, elbow, wrist)
            
            # curl counter logic
            if angle < 30:
                self.curl_stage = "up" 
            if angle > 140 and self.curl_stage == 'up':
                self.curl_stage = "down"  
                self.curl_counter +=1
            self.press_stage = None
            self.squat_stage = None
                
            # Viz joint angle
            self.viz_joint_angle(image, angle, elbow)
            
        elif self.current_action == 'press':           
            # Get coords
            shoulder = self.get_coordinates(landmarks, mp_pose, 'left', 'shoulder')
            elbow = self.get_coordinates(landmarks, mp_pose, 'left', 'elbow')
            wrist = self.get_coordinates(landmarks, mp_pose, 'left', 'wrist')

            # Calculate elbow angle
            elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
            
            # Compute distances between joints
            shoulder2elbow_dist = abs(math.dist(shoulder, elbow))
            shoulder2wrist_dist = abs(math.dist(shoulder, wrist))
            
            # Press counter logic
            if (elbow_angle > 130) and (shoulder2elbow_dist < shoulder2wrist_dist):
                self.press_stage = "up"
            if (elbow_angle < 50) and (shoulder2elbow_dist > shoulder2wrist_dist) and (self.press_stage == 'up'):
                self.press_stage = 'down'
                self.press_counter += 1
            self.curl_stage = None
            self.squat_stage = None
                
            # Viz joint angle
            self.viz_joint_angle(image, elbow_angle, elbow)
            
        elif self.current_action == 'squat':
            # Get coords
            # left side
            left_shoulder = self.get_coordinates(landmarks, mp_pose, 'left', 'shoulder')
            left_hip = self.get_coordinates(landmarks, mp_pose, 'left', 'hip')
            left_knee = self.get_coordinates(landmarks, mp_pose, 'left', 'knee')
            left_ankle = self.get_coordinates(landmarks, mp_pose, 'left', 'ankle')
            # right side
            right_shoulder = self.get_coordinates(landmarks, mp_pose, 'right', 'shoulder')
            right_hip = self.get_coordinates(landmarks, mp_pose, 'right', 'hip')
            right_knee = self.get_coordinates(landmarks, mp_pose, 'right', 'knee')
            right_ankle = self.get_coordinates(landmarks, mp_pose, 'right', 'ankle')
            
            # Calculate knee angles
            left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
            
            # Calculate hip angles
            left_hip_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
            right_hip_angle = self.calculate_angle(right_shoulder, right_hip, right_knee)
            
            # Squat counter logic
            thr = 165
            if (left_knee_angle < thr) and (right_knee_angle < thr) and (left_hip_angle < thr) and (right_hip_angle < thr):
                self.squat_stage = "down"
            if (left_knee_angle > thr) and (right_knee_angle > thr) and (left_hip_angle > thr) and (right_hip_angle > thr) and (self.squat_stage == 'down'):
                self.squat_stage = 'up'
                self.squat_counter += 1
            self.curl_stage = None
            self.press_stage = None
                
            # Viz joint angles
            self.viz_joint_angle(image, left_knee_angle, left_knee)
            self.viz_joint_angle(image, left_hip_angle, left_hip)
            self.viz_joint_angle(image, right_knee_angle, right_knee)
            self.viz_joint_angle(image, right_hip_angle, right_hip)
            
        else:
            pass
        return image

    def prob_viz(self, res, input_frame):
        """
        Displays the model prediction probability distribution over the set of exercise classes
        as a horizontal bar graph
        """
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):        
            cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), self.colors[num], -1)
            cv2.putText(output_frame, self.actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        return output_frame

    def process(self, image):
        """
        Processes the video frame from the user's webcam and runs the fitness trainer AI
        """
        try:
            # Pose detection model
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)

            # Draw the pose annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            image = self.draw_landmarks(image, results)
            
            # Prediction logic
            keypoints = self.extract_keypoints(results)
            self.sequence.append(keypoints.astype('float32'))
            self.sequence = self.sequence[-self.sequence_length:]

            if len(self.sequence) == self.sequence_length:
                res = model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
                
                self.current_action = self.actions[np.argmax(res)]
                confidence = np.max(res)
                
                # Erase current action variable if no probability is above threshold
                if confidence < self.threshold:
                    self.current_action = ''

                # Viz probabilities
                image = self.prob_viz(res, image)

                # Count reps
                try:
                    landmarks = results.pose_landmarks.landmark
                    image = self.count_reps(image, landmarks, mp_pose)
                except Exception as e:
                    logging.error(f"Error in counting reps: {e}")

                # Display graphical information
                cv2.rectangle(image, (0,0), (640, 40), self.colors[np.argmax(res)], -1)
                cv2.putText(image, f'curl {self.curl_counter}', (3,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f'press {self.press_counter}', (240,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f'squat {self.squat_counter}', (490,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
              
            return image

        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return image  # Return the original frame in case of error

def generate_frames(threshold1, threshold2, threshold3):
    """
    Generator function that yields video frames for streaming
    """
    processor = VideoProcessor(threshold1, threshold2, threshold3)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Resize frame for consistency
            frame = cv2.resize(frame, (640, 480))
            
            # Process frame
            processed_frame = processor.process(frame)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()

            # Yield frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """
    Home page
    """
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """
    Video streaming route
    """
    # Get thresholds from query parameters if provided
    t1 = request.args.get('threshold1', default=0.5, type=float)
    t2 = request.args.get('threshold2', default=0.5, type=float)
    t3 = request.args.get('threshold3', default=0.5, type=float)
    return Response(generate_frames(t1, t2, t3),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', debug=True)
    finally:
        cap.release()