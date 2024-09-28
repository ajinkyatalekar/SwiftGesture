import cv2
import streamlit as st
import numpy as np
import os

st.title("SwiftGesture")
st.text("Generate and run real-time gesture detection models in seconds! Train a new \nmodel, or run an exisiting one.")

st.subheader("Train a model")
st.text("Train a model to detect custom hand-landmarking gestures.")
button_train = st.button("Train")

st.subheader("Run a model")
st.text("Run a model to detect gestures.")
button_run = st.button("Run")