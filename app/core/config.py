# ad_api/app/core/config.py
# Centralized configuration for the Ad System Application

import logging
import os
import cv2
import numpy as np # For ULFD_PREPROC_MEAN

# --- Project Root Calculation ---
# config.py is in ad_api/app/core/
# SCRIPT_DIR will be ad_api/app/core/
# CORE_DIR (app/core) is SCRIPT_DIR
# APP_DIR (app/) is one level up from CORE_DIR
# PROJECT_ROOT (ad_api/) is one level up from APP_DIR
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(CORE_DIR)
PROJECT_ROOT = os.path.dirname(APP_DIR)

# --- Logging Configuration ---
LOG_LEVEL = logging.INFO  # logging.DEBUG for verbose, logging.INFO for production
LOG_FORMAT = '%(levelname)s - [%(threadName)s] - %(module)s.%(funcName)s: %(message)s'

# --- Model Paths (Absolute paths based on PROJECT_ROOT) ---
# Face Detector Model (Ultra-Light Fast Face Detector)
ULFD_ONNX_MODEL_PATH = os.path.join(PROJECT_ROOT, "face_model", "RFB-320wp.onnx")

# Age and Gender Classifier Models (GoogLeNet-based ONNX, for OpenCV DNN)
GENDER_MODEL_PATH_CV_DNN = os.path.join(PROJECT_ROOT, "gender_model", "gender_googlenet.onnx")
AGE_MODEL_PATH_CV_DNN = os.path.join(PROJECT_ROOT, "age_model", "age_googlenet.onnx")

# --- Face Detector Configuration (ULGFD) ---
ULFD_INPUT_SIZE_WH = [320, 240]  # [width, height] for network input
ULFD_CONF_THRESHOLD = 0.7       # Face confidence threshold
ULFD_IOU_THRESHOLD = 0.3        # NMS IOU threshold

# Preprocessing for ULFGFD (for cv2.dnn.blobFromImage)
ULFD_PREPROC_SCALEFACTOR = 1.0 / 128.0
ULFD_PREPROC_MEAN = (127.0, 127.0, 127.0) # BGR order

# Prior box generation parameters for ULFGFD
ULFD_PRIOR_CENTER_VARIANCE = 0.1
ULFD_PRIOR_SIZE_VARIANCE = 0.2
ULFD_PRIOR_MIN_BOXES = [[10.0, 16.0, 24.0], [32.0, 48.0], [64.0, 96.0], [128.0, 192.0, 256.0]]
ULFD_PRIOR_STRIDES = [8.0, 16.0, 32.0, 64.0]

# Expansion factor for face crops sent to Age/Gender classifiers
AG_CROP_EXPANSION_FACTOR = 0.25

# --- Age/Gender Classifier Configuration (GoogLeNet-based for OpenCV DNN) ---
AG_CLASSIFIER_INPUT_SIZE_WH = (224, 224) # (width, height) for classifier input
AG_CLASSIFIER_MEAN_VALS = (78.4263377603, 87.7689143744, 114.895847746) # BGR order

GENDER_LIST_CV_DNN = ['Male', 'Female'] # Output order for gender_googlenet.onnx
AGE_INTERVALS_CV_DNN = ['0-3', '4-7', '8-14', '15-24', '25-37', '38-47', '48-59', '60-100'] # For age_googlenet.onnx

# --- Ad System Configuration ---
ADS_DIRECTORY = os.path.join(PROJECT_ROOT, "ads")
DEFAULT_AD_DISPLAY_TIME_IMAGE = 10
MIN_AD_DISPLAY_TIME = 5
MIN_DETECTION_CONFIDENCE_FOR_TARGETED_ADS = ULFD_CONF_THRESHOLD
DETECTION_PERSISTENCE_SECONDS = 3.0
PREDICTION_WINDOW_DURATION_SECONDS = 2.5
PREDICTION_STABILITY_THRESHOLD_RATIO = 0.6
DEFAULT_AGE_FALLBACK = "25-37" # Adjusted to match one of the GoogLeNet intervals better

AGE_GROUP_DEFINITIONS = { # Max age (inclusive) for each group
    "child_max": 14,    # Covers '0-3', '4-7', '8-14' from GoogLeNet intervals
    "young_max": 37,    # Covers '15-24', '25-37'
    "adult_max": 59,    # Covers '38-47', '48-59'
}                       # 'senior' category handles '60-100'

# --- Main Application & Processing Service Configuration ---
# For PC testing, you can use higher values if your webcam/CPU supports it well.
# For RPi, these might be lower.
VIDEO_SOURCE = 0 # 0 for webcam, or "path/to/video.mp4"
WEBCAM_CAPTURE_WIDTH = 640
WEBCAM_CAPTURE_HEIGHT = 480
WEBCAM_FPS = 30 # Desired capture FPS (actual may vary)

FRAME_QUEUE_SIZE = 5 # Max frames buffered between capture and detector
DETECTION_RESULTS_QUEUE_SIZE = 5 # Max results buffered between detector and main app

DROP_FRAMES_IF_PROCESSING_BUSY = True

STATS_PRINT_INTERVAL_SECONDS = 5

# --- UI Display (OpenCV Window) Configuration ---
# This controls the local cv2.imshow() windows, independent of FastAPI streams.
SHOW_LOCAL_DISPLAY_WINDOW = True # Set to False if running headless or only want API output
LOCAL_DISPLAY_WINDOW_NAME = "Ad System - Local Display"
# For side-by-side view in the local OpenCV window:
LOCAL_DISPLAY_FEED_WIDTH = 480 # Width of the camera feed portion
LOCAL_DISPLAY_AD_MODE = 'side_by_side' # Options: 'side_by_side', 'on_feed' (ad on feed)

# Font settings for drawing on the local OpenCV display window
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_LABEL = 0.4
FONT_SCALE_CONF = 0.35
FONT_THICKNESS = 1

# --- FastAPI Server Configuration ---
API_HOST = "127.0.0.1"  # Host for FastAPI server (0.0.0.0 to be accessible on network)
API_PORT = 8000         # Port for FastAPI server
API_RELOAD = True       # For uvicorn --reload, used during development

# Stream FPS for MJPEG streams (approximate)
API_MJPEG_STREAM_FPS = 15 # Frames per second for the /feed_stream.mjpg and /ad_stream.mjpg
                          # Lower this if RPi struggles with encoding/streaming at high rates.