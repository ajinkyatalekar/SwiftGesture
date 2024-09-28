import cv2
import streamlit as st
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

from script import *

st.set_page_config(layout="wide")
st.title("SwiftGesture")
st.text("Generate and run real-time gesture recognition models in seconds! Train a new model, or run an exisiting one.")

# Running a model - Streamlit section
st.subheader("Run a model")
st.text("Run an existing model to detect gestures. (Make sure the model is in /models)")

model_path = file_selector()
st.write('You selected `%s`' % model_path)
button_run = st.button("Run")
run_frame_placeholder = st.empty()

# Running model
latest_detection_result = None
def result_callback(result, output_image, timestamp_ms):
    global latest_detection_result
    latest_detection_result = result

if button_run:
    # Handlandmarker detector setup
    base_options = python.BaseOptions(model_asset_path='src/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=result_callback
    )

    detector = vision.HandLandmarker.create_from_options(options)

    # Gesture detector setup
    gesture_detector = gesture_model(model_path)
    label,score = ('No hand detected', 0.0)

    cap = cv2.VideoCapture(0)

    button_run_stop = st.button("Stop")
    while (cap.isOpened and not button_run_stop):
        ret, frame = cap.read()

        if not ret:
            break

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp_ms = int(time.time() * 1000)
        detector.detect_async(image, timestamp_ms=timestamp_ms)

        if latest_detection_result:
            label, score = gesture_detector.predict(latest_detection_result, 0.5)
            annotated_image = draw_landmarks_on_image(frame, latest_detection_result, label, score)
        else:
            annotated_image = frame

        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        run_frame_placeholder.image(annotated_image, channels="RGB")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


st.subheader("OR")


# Training a model - Streamlit section
st.subheader("Train a model")
st.text("Train your own model to detect custom hand-landmarking gestures.")

# Model name
model_name_input = st.text_input("What would you like to name the model?")
st.write('Your model will be saved at: `models/%s`' % model_name_input)

# Gesture names
gesture_list_input = st.text_input("Enter gesture names as a comma separated list.", "fist,palm")
gesture_list_input = gesture_list_input.replace(" ", "").split(",")
st.write(gesture_list_input if len(gesture_list_input) >= 1 and gesture_list_input[0]!="" else "")

button_train = st.button("Begin Training")
train_frame_placeholder = st.empty()


# Training model
if button_train:
    button_abort = st.button("Abort")
    trainer = gesture_model_trainer()
    trainer.run_train_loop(gesture_list_input, train_frame_placeholder)