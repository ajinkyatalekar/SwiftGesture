import streamlit as st
import os
import json
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import cv2
import mediapipe as mp
import time

# Streamlit webapp functionality
def file_selector(folder_path='models'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a model', filenames)
    return os.path.join(folder_path, selected_filename)

# Visualization
def draw_landmarks_on_image(rgb_image, detection_result, label, score = ""):
    MARGIN = 30
    FONT_SIZE = 1
    FONT_THICKNESS = 2
    HANDEDNESS_TEXT_COLOR = (88, 205, 54)

    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
        annotated_image,
        hand_landmarks_proto,
        solutions.hands.HAND_CONNECTIONS,
        solutions.drawing_styles.get_default_hand_landmarks_style(),
        solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Add annotations
        text_str = f"{handedness[0].category_name} Adding {label}"
        if score:
            text_str = f"{label} {handedness[0].category_name} {score:.2f}"

        cv2.putText(annotated_image, text_str,
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

# Running a model
class gesture_model:
    def __init__(self, model_path):
        self.loaded_model = load_model(f'{model_path}/model.keras')
        with open(f'{model_path}/label_map.json', 'r') as f:
            file = json.load(f)
            self.all_labels = list(file.values())

    def predict(self,detection_result, threshold):
        out_landmarks = []
        if (detection_result.hand_world_landmarks):
            change = 1
            if (detection_result.handedness[0][0].category_name == "Left"):
                change = -1

            for hand in detection_result.hand_world_landmarks:
                for landmark in hand:
                    out_landmarks.append([landmark.x * change, landmark.y, landmark.z])

        if len(out_landmarks) == 0:
            return ('No hand detected', 0.0)

        out_landmarks = np.array(out_landmarks)
        out_landmarks = np.expand_dims(out_landmarks, axis=0)
        prediction = self.loaded_model.predict(out_landmarks, verbose=0)

        if (prediction.max() < threshold):
                return ('Unknown', 0.0)

        return (self.all_labels[np.argmax(prediction)], prediction.max())


# Training a model
class gesture_model_trainer:
    def __init__(self):
        pass

    # add detection and current_gesture to a variable that can be exported to a json file
    def add_detection_object(self, base, current_gesture, detection):
        if not detection.hand_world_landmarks:
            return
            
        landmarks = []
        for landmark in detection.hand_world_landmarks[0]:
            landmarks.append([landmark.x, landmark.y, landmark.z])
        data = {
            "gesture": current_gesture,
            "landmarks": landmarks
        }
        base["data"].append(data)

    def run_train_loop(self, gestures, train_frame_placeholder):
        gesture_index = 0

        self.latest_detection_result = None
        def result_callback(result, output_image, timestamp_ms):
            self.latest_detection_result = result

        base_options = python.BaseOptions(model_asset_path='src/hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=result_callback
        )
        detector = vision.HandLandmarker.create_from_options(options)

        # Creating variables for database
        base = {
            "data": [],
            "labels": gestures
        }

        cap = cv2.VideoCapture(0)

        while gesture_index < len(gestures):
            ret, frame = cap.read()
            if not ret:
                break

            current_gesture = gestures[gesture_index]

            # STEP 3: Load the input image.
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            timestamp_ms = int(time.time() * 1000)

            # Perform detection asynchronously
            detector.detect_async(image, timestamp_ms=timestamp_ms)

            # Draw landmarks on the image
            if self.latest_detection_result:
                annotated_image = draw_landmarks_on_image(frame, self.latest_detection_result, current_gesture)
            else:
                annotated_image = frame

            # Display the resulting frame
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            train_frame_placeholder.image(annotated_image, channels="RGB")

            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key:
                if key == ord(' '): # Space to next image
                    obj = add_detection_object(base, current_gesture, self.latest_detection_result)
                    
                elif key == 13: # Enter to next gesture
                    gesture_index += 1
            

        cap.release()
        cv2.destroyAllWindows()