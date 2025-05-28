# config.py
# Centralized configuration for the Raspberry Pi Ad System Application

import logging
import os
import numpy as np # For ULFD_PREPROC_MEAN
import cv2

# --- Project Root Calculation ---
# Assumes config.py is in the 'main' subdirectory of the project.
# SCRIPT_DIR is .../RaspAd_V5/main/
# PROJECT_ROOT is .../RaspAd_V5/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# --- Logging Configuration ---
LOG_LEVEL = logging.INFO  # logging.DEBUG for verbose, logging.INFO for production
LOG_FORMAT = '%(asctime)s - %(levelname)s - [%(threadName)s] - %(module)s.%(funcName)s: %(message)s'

# --- Model Paths (Absolute paths based on PROJECT_ROOT) ---
# Face Detector Model (Ultra-Light Fast Face Detector)
ULFD_ONNX_MODEL_PATH = os.path.join(PROJECT_ROOT, "face_model", "RFB-320wp.onnx")

# Age and Gender Classifier Models (GoogLeNet-based ONNX, for OpenCV DNN)
GENDER_MODEL_PATH_CV_DNN = os.path.join(PROJECT_ROOT, "gender_model", "gender_googlenet.onnx")
AGE_MODEL_PATH_CV_DNN = os.path.join(PROJECT_ROOT, "age_model", "age_googlenet.onnx")

# --- Face Detector Configuration (ULGFD) ---
ULFD_INPUT_SIZE_WH = [320, 240]  # [width, height] for network input
ULFD_CONF_THRESHOLD = 0.7       # Confidence threshold for detected faces
ULFD_IOU_THRESHOLD = 0.3        # NMS IOU threshold

# Preprocessing for ULFGFD (for cv2.dnn.blobFromImage)
ULFD_PREPROC_SCALEFACTOR = 1.0 / 128.0
ULFD_PREPROC_MEAN = (127.0, 127.0, 127.0) # BGR order: (blue_mean, green_mean, red_mean)

# Prior box generation parameters for ULFGFD (must match the model training)
ULFD_PRIOR_CENTER_VARIANCE = 0.1
ULFD_PRIOR_SIZE_VARIANCE = 0.2
ULFD_PRIOR_MIN_BOXES = [[10.0, 16.0, 24.0], [32.0, 48.0], [64.0, 96.0], [128.0, 192.0, 256.0]]
ULFD_PRIOR_STRIDES = [8.0, 16.0, 32.0, 64.0]

# Expansion factor for face crops sent to Age/Gender classifiers
# (e.g., 0.2 means expand width/height by 20% from center)
AG_CROP_EXPANSION_FACTOR = 0.25

# --- Age/Gender Classifier Configuration (GoogLeNet-based for OpenCV DNN) ---
AG_CLASSIFIER_INPUT_SIZE_WH = (224, 224) # (width, height) for classifier input
AG_CLASSIFIER_MEAN_VALS = (78.4263377603, 87.7689143744, 114.895847746) # BGR order

# Labels MUST match the output order and content of your GoogLeNet models
GENDER_LIST_CV_DNN = ['Male', 'Female'] # If class 0 is Male, class 1 is Female
AGE_INTERVALS_CV_DNN = ['0-3', '4-7', '8-14', '15-24', '25-37', '38-47', '48-59', '60-100']

# --- Ad System Configuration ---
ADS_DIRECTORY = os.path.join(PROJECT_ROOT, "ads")
DEFAULT_AD_DISPLAY_TIME_IMAGE = 10  # Seconds for static image ads
MIN_AD_DISPLAY_TIME = 5             # Min seconds any ad (image/video) must show
MIN_DETECTION_CONFIDENCE_FOR_TARGETED_ADS = ULFD_CONF_THRESHOLD # Ad system uses face conf
DETECTION_PERSISTENCE_SECONDS = 3.0   # How long a detection "lingers" for ad targeting
PREDICTION_WINDOW_DURATION_SECONDS = 2.5 # Stability window for demographic category voting
PREDICTION_STABILITY_THRESHOLD_RATIO = 0.6 # Higher ratio means stronger consensus needed (e.g., 0.6 for 60%)

DEFAULT_AGE_FALLBACK = "20-29" # Fallback age interval if detection is unclear
# Age group definitions for mapping detected age intervals to ad categories
AGE_GROUP_DEFINITIONS = { # Max age (inclusive) for each group
    "child_max": 14,    # e.g., utils.txt '8-14' maps to child. '15-24' maps to young.
    "young_max": 37,    # e.g., '25-37' maps to young. '38-47' maps to adult.
    "adult_max": 59,    # e.g., '48-59' maps to adult. '60-100' maps to senior.
}                       # 'senior' category handles ages > adult_max

# --- Main Application & Display Configuration ---
VIDEO_SOURCE = 0 # 0 for webcam, or "path/to/video.mp4"
WEBCAM_CAPTURE_WIDTH = 640
WEBCAM_CAPTURE_HEIGHT = 480
WEBCAM_FPS = 15 # Desired capture FPS (actual may vary). Lower for RPi if CPU is bottlenecked by capture.

FRAME_QUEUE_SIZE = 3 # Max frames buffered between capture and detector. Small for low latency.
DETECTION_RESULTS_QUEUE_SIZE = 3 # Max results buffered between detector and main app.

DROP_FRAMES_IF_PROCESSING_BUSY = True # If True, newer frames preferred over processing all.

STATS_PRINT_INTERVAL_SECONDS = 5 # How often to print FPS/Queue stats

# UI Display settings
SHOW_COMBINED_OUTPUT_ON_SCREEN = False
SHOW_ADS_ON_SCREEN = True # If not combined, should ads be shown in their own window?
COMBINED_OUTPUT_WINDOW_NAME = "Ad System Display"
AD_DISPLAY_WINDOW_WIDTH = 1280 # Target width for the ad part of the display
AD_DISPLAY_WINDOW_HEIGHT = 720 # Target height for the ad part of the display

# For side-by-side view, how wide should the camera feed part be?
COMBINED_VIEW_FEED_WIDTH = 320  # Smaller feed view for less processing if resized for display
# FONT settings for drawing on screen
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_LABEL = 0.4
FONT_SCALE_CONF = 0.35
FONT_THICKNESS = 1

AD_DISPLAY_MODE_ON_COMBINED_VIEW = 'side_by_side' # Options: 'side_by_side', 'on_feed' (less practical for separate ad)