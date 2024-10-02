# SwiftGesture
Python app to create and use hand gesture classification models.

## What It Does
SwiftGesture combines a hand landmark detection model with a gesture detection model to classify custom gestures in real time. SwiftGesture primarily focuses on creating models blazingly fast with the least input data needed. You can train the model to classify the American Sign Language with over 95% accuracy using less than 80 images.

## Getting Started

### Prerequisites
- Install dependencies: run `pip install -r requirements.txt`
- On Linux: run `apt-get install libgl1`

### Usage
- Start the app using `streamlit run path/to/home.py`
- That's it! Now you can train a model or run an existing one using the buttons on the Streamlit app at `http://localhost:8501`.
