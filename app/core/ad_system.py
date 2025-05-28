# ad_system.py

import cv2
import numpy as np
import time
import os
from threading import Thread, Lock, Event
from collections import Counter
import logging

# Import configurations
import config

logger = logging.getLogger(__name__)

class AdSystem:
    """
    Manages and displays advertisements based on demographic information.
    """
    
    def __init__(self):
        """Initializes the advertisement system."""
        logger.info("Initializing AdSystem...")
        
        # Load configurations
        self.ads_directory = config.ADS_DIRECTORY
        self.default_display_time_image = config.DEFAULT_AD_DISPLAY_TIME_IMAGE
        self.min_display_time = config.MIN_AD_DISPLAY_TIME
        # self.transition_time = config.TRANSITION_TIME_SECONDS # Placeholder
        self.min_detection_confidence = config.MIN_DETECTION_CONFIDENCE_FOR_TARGETED_ADS
        self.detection_persistence = config.DETECTION_PERSISTENCE_SECONDS
        self.display_size = (config.AD_DISPLAY_WINDOW_WIDTH, config.AD_DISPLAY_WINDOW_HEIGHT)
        self.prediction_window_duration = config.PREDICTION_WINDOW_DURATION_SECONDS
        self.prediction_stability_threshold_ratio = config.PREDICTION_STABILITY_THRESHOLD_RATIO
        self.age_group_definitions = config.AGE_GROUP_DEFINITIONS
        self.default_age_fallback = config.DEFAULT_AGE_FALLBACK
        
        self.ad_categories = {
            "default": [], "general": [],
            "male": {"child": [], "young": [], "adult": [], "senior": []},
            "female": {"child": [], "young": [], "adult": [], "senior": []}
        }
        
        # State tracking
        self.last_faces_info = [] # Stores the raw faces_info from the detector
        self.last_detection_time = 0 # Timestamp of the last valid detection input
        self.current_ad_info = None # Dict: {"type": "image/video", "path": "..."}
        self.ad_start_time = 0 # Timestamp when the current_ad_info started displaying
        
        self.ad_lock = Lock() # Protects shared ad state (playlist, category, current_ad_info)
        self.playback_thread = None
        self.stop_event = Event() # Signals the playback thread to stop
        
        # Current frame for display (shared with the main thread)
        self.current_ad_frame = np.zeros((self.display_size[1], self.display_size[0], 3), dtype=np.uint8)
        self.frame_lock = Lock() # Protects current_ad_frame
        
        # Playlist management
        self.current_playlist = []
        self.current_playlist_index = 0
        self.current_category_playing = "default" # The category whose playlist is currently active
        self.requested_category_target = "default" # The category determined by latest detections
        
        # Video playback state
        self.video_cap = None
        self.current_video_path_playing = None # Path of the video currently loaded in video_cap
        
        # Prediction stability tracking
        self.prediction_history = [] # List of (timestamp, determined_category) tuples
        self.last_stable_category_voted = "default" # The most recent stable category from voting
        
        # Caches and pre-allocated buffers
        self._image_cache = {} # Cache for loaded and resized image ads
        self._resized_video_frame_buffer = np.zeros((self.display_size[1], self.display_size[0], 3), dtype=np.uint8)

        self._load_advertisements()
        
        # Initialize playlist if no specific category is active yet
        if not self.current_playlist and self.ad_categories["default"]:
             logger.info("Initializing playlist with default ads as no category was pre-selected.")
             self.current_playlist = list(self.ad_categories["default"]) # Use a copy
             self.current_category_playing = "default"
        elif not self.current_playlist:
            logger.warning("No default ads found to initialize playlist. Ad screen may remain blank.")

        self.playback_thread = Thread(target=self._playback_loop, name="AdPlaybackThread", daemon=True)
        self.playback_thread.start()
        logger.info("AdSystem initialized and playback thread started.")

    def _load_advertisements(self):
        logger.info(f"Loading advertisements from directory: {self.ads_directory}")
        if not os.path.exists(self.ads_directory):
            logger.warning(f"Ads directory '{self.ads_directory}' not found. Creating it.")
            os.makedirs(self.ads_directory, exist_ok=True)
            # Create subdirectories based on ad_categories structure
            for category in ["default", "general"]:
                os.makedirs(os.path.join(self.ads_directory, category), exist_ok=True)
            for gender in self.ad_categories["male"]: # Iterate over defined gender keys
                os.makedirs(os.path.join(self.ads_directory, gender), exist_ok=True)
                for age_group in self.ad_categories[gender]: # Iterate over defined age_group keys
                    os.makedirs(os.path.join(self.ads_directory, gender, age_group), exist_ok=True)
        
        # Load ads for each category
        self.ad_categories["default"] = self._load_ads_from_dir(os.path.join(self.ads_directory, "default"))
        self.ad_categories["general"] = self._load_ads_from_dir(os.path.join(self.ads_directory, "general"))
        
        for gender in ["male", "female"]:
            for age_group in ["child", "young", "adult", "senior"]:
                path = os.path.join(self.ads_directory, gender, age_group)
                self.ad_categories[gender][age_group] = self._load_ads_from_dir(path)
        
        if not self.ad_categories["default"]:
            logger.warning("CRITICAL: No default ads found! System needs default ads to function when no one is detected.")
        if not self.ad_categories["general"] and any(not self.ad_categories[g][a] for g in ["male", "female"] for a in self.ad_categories[g]):
            logger.warning("No general ads found. System will rely on default or highly specific ads.")
        
        # Initialize current_playlist with default if it's empty
        if not self.current_playlist and self.ad_categories["default"]:
            self.current_playlist = list(self.ad_categories["default"])
            self.current_category_playing = "default"
            logger.info(f"Initialized with default playlist: {len(self.current_playlist)} ads.")


    def _load_ads_from_dir(self, directory_path: str) -> list:
        ads = []
        if not os.path.exists(directory_path):
            logger.warning(f"Ad subdirectory '{directory_path}' not found.")
            return ads
            
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        
        try:
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                if os.path.isfile(file_path):
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in image_extensions:
                        ads.append({"type": "image", "path": file_path})
                    elif ext in video_extensions:
                        ads.append({"type": "video", "path": file_path})
            logger.info(f"Loaded {len(ads)} ads from {directory_path}")
        except OSError as e:
            logger.error(f"Could not list directory {directory_path}: {e}")
        return ads

    def _map_age_to_group(self, age_interval_str: str) -> str:
        """Maps an age interval string (e.g., '20-29', '60+') to a defined age group."""
        try:
            # Simplistic parsing, can be made more robust
            if '-' in age_interval_str:
                parts = age_interval_str.split('-')
                min_age = int(parts[0])
                max_age = int(parts[1])
                # Use midpoint or min_age for categorization
                effective_age = (min_age + max_age) / 2 
            elif '+' in age_interval_str:
                effective_age = int(age_interval_str.replace('+', ''))
            else: # Assume single age
                effective_age = int(age_interval_str)

            if effective_age <= self.age_group_definitions["child_max"]: return "child"
            if effective_age <= self.age_group_definitions["young_max"]: return "young"
            if effective_age <= self.age_group_definitions["adult_max"]: return "adult"
            return "senior"
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not parse age interval '{age_interval_str}': {e}. Defaulting to 'young'.")
            return "young" # Fallback for parsing errors
    
    def _select_playlist_for_category(self, target_category_name: str):
        """
        Sets the current playlist based on the target category name, with fallbacks.
        This method should be called with self.ad_lock held.
        """
        logger.debug(f"Attempting to switch to category: {target_category_name}")
        new_playlist = []
        
        # Determine primary playlist
        if target_category_name == "default":
            new_playlist = self.ad_categories["default"]
        elif target_category_name == "general":
            new_playlist = self.ad_categories["general"]
        elif "_" in target_category_name: # e.g., "male_young"
            gender, age_g = target_category_name.split("_", 1)
            if gender in self.ad_categories and age_g in self.ad_categories[gender]:
                new_playlist = self.ad_categories[gender][age_g]
            
        # Apply fallbacks if primary playlist is empty or doesn't exist
        final_category_name_for_playlist = target_category_name
        if not new_playlist:
            logger.info(f"No ads for '{target_category_name}', trying 'general'.")
            new_playlist = self.ad_categories["general"]
            final_category_name_for_playlist = "general"
        if not new_playlist: # Try default if general also fails
            logger.info(f"No ads for '{target_category_name}' or 'general', trying 'default'.")
            new_playlist = self.ad_categories["default"]
            final_category_name_for_playlist = "default"
        
        if not new_playlist:
            logger.warning(f"No ads found for '{target_category_name}' or any fallbacks. Playlist will be empty.")
            # Keep the current playlist and category if all fallbacks fail to find ads,
            # or clear it if that's the desired behavior.
            # For now, let's clear it to signify no suitable ads.
            self.current_playlist = []
        else:
            self.current_playlist = list(new_playlist) # Use a copy for modification safety

        self.current_category_playing = final_category_name_for_playlist
        self.current_playlist_index = 0
        if self.video_cap: # Release video if category changes
            self.video_cap.release()
            self.video_cap = None
            self.current_video_path_playing = None
        
        logger.info(f"Switched to playlist for category '{self.current_category_playing}' with {len(self.current_playlist)} ads.")


    def _get_next_ad_from_current_playlist(self):
        """
        Gets the next ad from the current_playlist. Call with self.ad_lock held.
        """
        if not self.current_playlist:
            return None
        
        ad_info = self.current_playlist[self.current_playlist_index]
        self.current_playlist_index = (self.current_playlist_index + 1) % len(self.current_playlist)
        return ad_info
        
    def _playback_loop(self):
        logger.info("Ad playback loop started.")
        last_ad_event_time = time.monotonic() # Time of last ad change or video end
        
        while not self.stop_event.is_set():
            time.sleep(0.01) # Main loop throttle, adjust if needed
            current_time = time.monotonic()
            
            with self.ad_lock:
                # --- 1. Check if category needs to change ---
                requested_target_is_different = (self.requested_category_target != self.current_category_playing)
                min_time_for_current_ad_passed = (current_time - self.ad_start_time >= self.min_display_time)

                if requested_target_is_different and min_time_for_current_ad_passed:
                    self._select_playlist_for_category(self.requested_category_target)
                    self.current_ad_info = None # Force new ad selection from new playlist
                    last_ad_event_time = current_time # Playlist changed, treat as an ad event

                # --- 2. Check if current ad needs to be changed (e.g., image timeout) ---
                needs_new_ad_selection = self.current_ad_info is None

                if self.current_ad_info and self.current_ad_info["type"] == "image" and \
                   (current_time - self.ad_start_time >= self.default_display_time_image):
                    needs_new_ad_selection = True
                    logger.debug("Image ad display time elapsed.")
                
                # --- 3. Select new ad if needed ---
                if needs_new_ad_selection:
                    if not self.current_playlist:
                        logger.debug(f"Current playlist for '{self.current_category_playing}' is empty. No ad to select.")
                        self.current_ad_info = None # Ensure it's None
                    else:
                        self.current_ad_info = self._get_next_ad_from_current_playlist()
                        if self.current_ad_info:
                             logger.info(f"Playing next ad: {self.current_ad_info['path']}")
                             self.ad_start_time = current_time # New ad starts now
                             last_ad_event_time = current_time # Mark this as the last ad event
                             # If it's a new video, video_cap will be opened by _get_frame_for_ad
                        else:
                            logger.error("Failed to get next ad from a non-empty playlist. This shouldn't happen.")
                            # Potentially reset playlist or handle error
            
            # --- 4. Get frame for the current ad and handle video ending ---
            if self.current_ad_info:
                frame_to_display, video_just_ended = self._get_frame_for_ad(self.current_ad_info)
                
                if video_just_ended: # This means a video ad has completed its playback
                    logger.info(f"Video ad ended: {self.current_ad_info['path']}")
                    with self.ad_lock: # Lock needed to modify current_ad_info
                        self.current_ad_info = None # Force selection of a new ad in next loop iteration
                        last_ad_event_time = current_time # Mark video end as an ad event
                
                with self.frame_lock:
                    self.current_ad_frame = frame_to_display
            else: # No ad currently selected (e.g., empty playlist)
                with self.frame_lock:
                    self.current_ad_frame = np.zeros((self.display_size[1], self.display_size[0], 3), dtype=np.uint8)
        
        logger.info("Ad playback loop stopped.")
    
    def _get_frame_for_ad(self, ad_info_dict: dict):
        """
        Gets a frame for the given ad. Handles image caching and video playback.
        Returns: (frame, video_ended_flag)
        video_ended_flag is True if a video ad has finished playing its content.
        """
        if not ad_info_dict:
            return np.zeros((self.display_size[1], self.display_size[0], 3), dtype=np.uint8), False
            
        path = ad_info_dict["path"]
        if ad_info_dict["type"] == "image":
            try:
                if path not in self._image_cache:
                    image = cv2.imread(path)
                    if image is None:
                        logger.error(f"Failed to load image ad: {path}")
                        self._image_cache[path] = "ERROR" # Mark as bad to avoid retrying constantly
                        return np.zeros((self.display_size[1], self.display_size[0], 3), dtype=np.uint8), True # Treat as ended
                    self._image_cache[path] = cv2.resize(image, self.display_size)
                
                if self._image_cache[path] == "ERROR":
                    return np.zeros((self.display_size[1], self.display_size[0], 3), dtype=np.uint8), True

                return self._image_cache[path], False 
            except Exception as e:
                logger.exception(f"Error processing image ad {path}: {e}")
                self._image_cache[path] = "ERROR"
                return np.zeros((self.display_size[1], self.display_size[0], 3), dtype=np.uint8), True
        
        elif ad_info_dict["type"] == "video":
            path = ad_info_dict["path"] # Get path here
            try:
                # Open video if it's new, cap is closed, or path changed
                if self.current_video_path_playing != path or self.video_cap is None or not self.video_cap.isOpened():
                    if self.video_cap: self.video_cap.release()
                    
                    # --- GStreamer Pipeline Attempt ---
                    # Ensure GStreamer plugins are installed on RPi:
                    # sudo apt install gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-omx-rpi (or v4l2 equivalents)
                    
                    # # Option 1: Using decodebin (tries to auto-configure)
                    # gst_pipeline_str = (
                    #     f"filesrc location=\"{path}\" ! "
                    #     f"decodebin ! " 
                    #     f"videoconvert ! " 
                    #     f"video/x-raw,format=BGR ! " 
                    #     f"appsink max-buffers=2 drop=true emit-signals=true" # emit-signals might help with responsiveness
                    # )

                    # Option 2: Explicit H.264 decoding (more specific, might be needed)
                    # Check which decoder element works on your RPi OS version (omxh264dec or v4l2h264dec)
                    # For newer RPi OS (Bullseye/Bookworm with libcamera), v4l2h264dec is more likely.
                    gst_pipeline_str = (
                        f"filesrc location=\"{path}\" ! "
                        f"qtdemux ! h264parse ! v4l2h264dec ! " # or omxh264dec
                        f"videoconvert ! "
                        f"video/x-raw,format=BGR ! "
                        f"appsink max-buffers=2 drop=true emit-signals=true"
                    )
                    # If your videos are reliably H.264, you might try gst_pipeline_str_h264 first.
                    # For now, let's stick with decodebin for broader compatibility.

                    logger.debug(f"Attempting to open video with GStreamer pipeline: {gst_pipeline_str}")
                    self.video_cap = cv2.VideoCapture(gst_pipeline_str, cv2.CAP_GSTREAMER)
                    
                    if not self.video_cap.isOpened():
                        logger.warning(f"Failed to open with GStreamer: {path}. Falling back to default OpenCV open.")
                        self.video_cap = cv2.VideoCapture(path) # Fallback
                        if not self.video_cap.isOpened():
                             logger.error(f"Could not open video even with fallback: {path}")
                             self.current_video_path_playing = None
                             # Return blank frame and True for video_ended (as it failed to open)
                             return np.zeros((self.display_size[1], self.display_size[0], 3), dtype=np.uint8), True
                    else:
                        logger.info(f"Successfully opened video with GStreamer: {path}")

                    self.current_video_path_playing = path
                
                ret, frame = self.video_cap.read()
                if not ret or frame is None: 
                    logger.info(f"Video {path} (GStreamer/fallback): end of stream or read error.")
                    if self.video_cap: self.video_cap.release()
                    self.video_cap = None
                    self.current_video_path_playing = None
                    return np.zeros((self.display_size[1], self.display_size[0], 3), dtype=np.uint8), True 
                
                # If GStreamer pipeline includes videoscale to self.display_size, this resize might be redundant or harmful.
                # For now, assume GStreamer pipeline provides frames at native video res, and we resize.
                cv2.resize(frame, self.display_size, dst=self._resized_video_frame_buffer)
                return self._resized_video_frame_buffer, False 
            except Exception as e:
                logger.exception(f"Error playing video ad {path}: {e}")
                if self.video_cap: self.video_cap.release()
                self.video_cap = None
                self.current_video_path_playing = None
                return np.zeros((self.display_size[1], self.display_size[0], 3), dtype=np.uint8), True


    def _get_voted_stable_category(self, current_timestamp: float, raw_detected_category: str) -> str:
        """Determines the most stable category based on recent detections."""
        # Remove outdated predictions
        window_start_time = current_timestamp - self.prediction_window_duration
        self.prediction_history = [(t, c) for t, c in self.prediction_history if t >= window_start_time]
        
        self.prediction_history.append((current_timestamp, raw_detected_category))
        
        if not self.prediction_history: # Should not be empty after append
            return self.last_stable_category_voted 
            
        category_counts = Counter(cat for _, cat in self.prediction_history)
        most_common_category, count = category_counts.most_common(1)[0]
        
        required_vote_count = len(self.prediction_history) * self.prediction_stability_threshold_ratio
        
        if count >= required_vote_count:
            if self.last_stable_category_voted != most_common_category:
                logger.info(f"Voting stabilized category to: {most_common_category} (votes: {count}/{len(self.prediction_history)})")
            self.last_stable_category_voted = most_common_category
            return most_common_category
        
        return self.last_stable_category_voted # Stick to last stable if not enough consensus


    def update_target_audience(self, faces_info_from_detector: list):
        """
        Processes face detection results to determine the target ad category.
        This method is called by the main application thread.
        """
        current_time = time.monotonic()
        raw_determined_category = "default" 
        
        if faces_info_from_detector: # Check if the list itself is not empty
            # Filter out faces with very low confidence even before persistence check,
            # or ensure the detector itself does this.
            # For simplicity here, assume faces_info_from_detector contains reasonably confident faces.
            self.last_faces_info = faces_info_from_detector
            self.last_detection_time = current_time
        
        # Check if there's a recent, valid detection
        has_recent_valid_audience = (current_time - self.last_detection_time) < self.detection_persistence and self.last_faces_info
        
        if not has_recent_valid_audience:
            raw_determined_category = "default"
            logger.debug("No recent valid audience, category will be: default")
        else:
            # Analyze self.last_faces_info
            if len(self.last_faces_info) > 1:
                raw_determined_category = "general"
                logger.debug(f"Multiple faces ({len(self.last_faces_info)}), category will be: general")
            else: # Single person in self.last_faces_info
                face = self.last_faces_info[0]
                # Confidence check from detector's output for this specific face
                face_confidence = face.get("confidence", 0.0)
                
                if face_confidence < self.min_detection_confidence:
                    raw_determined_category = "general"
                    logger.debug(f"Single face confidence ({face_confidence:.2f}) too low, category will be: general")
                else:
                    gender = face.get("gender", "Unknown").lower()
                    age_interval = face.get("age", self.default_age_fallback) 
                    
                    if gender == "unknown" or age_interval == "unknown" or age_interval == self.default_age_fallback and gender == "unknown":
                        raw_determined_category = "general"
                        logger.debug(f"Demographics are Unknown (gender={gender}, age={age_interval}), category will be: general")
                    else:
                        age_group = self._map_age_to_group(age_interval)
                        raw_determined_category = f"{gender}_{age_group}"
                        logger.debug(f"Targeted audience: {gender} {age_group} (from {age_interval}), category will be: {raw_determined_category}")
        
        # Apply voting for stability
        voted_stable_cat = self._get_voted_stable_category(current_time, raw_determined_category)
        
        with self.ad_lock: # Protect write to requested_category_target
            if self.requested_category_target != voted_stable_cat:
                logger.info(f"Updating requested ad target category from '{self.requested_category_target}' to '{voted_stable_cat}'.")
            self.requested_category_target = voted_stable_cat

    def get_current_ad_frame(self) -> np.ndarray:
        """Returns a copy of the current ad frame for display."""
        with self.frame_lock:
            return self.current_ad_frame.copy()

    def cleanup(self):
        logger.info("Cleaning up AdSystem...")
        self.stop_event.set() # Signal playback thread to stop
        if self.playback_thread and self.playback_thread.is_alive():
            logger.info("Waiting for ad playback thread to join...")
            self.playback_thread.join(timeout=2.0)
            if self.playback_thread.is_alive():
                logger.warning("Ad playback thread did not join in time.")
        
        if self.video_cap:
            logger.info("Releasing video capture for ads.")
            self.video_cap.release()
        
        # Clear caches if necessary, though Python's GC will handle it on exit
        self._image_cache.clear()
        logger.info("AdSystem cleanup complete.")

# Example self-test (optional, for testing this module independently)
if __name__ == '__main__':
    # Configure logging for the self-test
    # Ensure config.py paths for ADS_DIRECTORY are correct (e.g., "../ads")
    # if running this script directly from the 'main' subdirectory.
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
    logger.info("Running AdSystem self-test...")
    logger.info(f"Using ADS_DIRECTORY from config: {config.ADS_DIRECTORY}")

    if not os.path.exists(config.ADS_DIRECTORY) or not os.listdir(config.ADS_DIRECTORY):
        logger.error(f"Actual ads directory '{config.ADS_DIRECTORY}' not found or is empty.")
        logger.error("Please ensure the ads directory is correctly populated and path in config.py is correct.")
        logger.error("Aborting AdSystem self-test.")
        exit() # Or raise an error

    ad_sys = None
    try:
        ad_sys = AdSystem() # This will now use the ADS_DIRECTORY from config.py
        logger.info("AdSystem initialized for self-test using configured ads directory.")
        
        # Test 1: Default ads
        logger.info("--- Test 1: Simulating 'default' state (no detections) ---")
        ad_sys.update_target_audience([]) # Empty list means no one detected
        logger.info(f"Waiting for AdSystem to play default ads (min_display_time: {ad_sys.min_display_time}s)...")
        # Allow time for at least one ad to potentially play from the default category
        time.sleep(ad_sys.min_display_time + 1) 
        
        current_frame = ad_sys.get_current_ad_frame()
        logger.info(f"Ad frame shape after default: {current_frame.shape}")
        with ad_sys.ad_lock: # Access shared state safely
            logger.info(f"Currently playing category: {ad_sys.current_category_playing}")
            logger.info(f"Current ad info: {ad_sys.current_ad_info}")
            # We can't be sure 'default' is the category if it has no ads,
            # but it should at least try to load it.
            if not ad_sys.ad_categories["default"]:
                 logger.warning("No ads in 'default' category to assert against.")
            else:
                assert ad_sys.current_category_playing == "default", \
                       f"Expected default, got {ad_sys.current_category_playing}"


        # Test 2: Targeted ads (e.g., male_child)
        logger.info("--- Test 2: Simulating 'male_child' detection ---")
        dummy_male_child_info = [{'box_original': [0,0,10,10], 'gender': 'Male', 'age': '0-12', 'confidence': 0.9}]
        
        num_updates = int(config.PREDICTION_WINDOW_DURATION_SECONDS / 0.2) + 5
        logger.info(f"Sending {num_updates} 'male_child' detection updates over ~{num_updates*0.2:.1f}s...")
        for i in range(num_updates):
            ad_sys.update_target_audience(dummy_male_child_info)
            time.sleep(0.2) 
        
        logger.info(f"Waiting for AdSystem to switch and play targeted ad (min_display_time: {ad_sys.min_display_time}s)...")
        time.sleep(ad_sys.min_display_time + 1)
        
        current_frame = ad_sys.get_current_ad_frame()
        logger.info(f"Ad frame shape after male_child detection: {current_frame.shape}")
        with ad_sys.ad_lock:
            logger.info(f"Requested target category: {ad_sys.requested_category_target}")
            logger.info(f"Last stable voted category: {ad_sys.last_stable_category_voted}")
            logger.info(f"Currently playing category: {ad_sys.current_category_playing}")
            logger.info(f"Current ad info: {ad_sys.current_ad_info}")
            if not ad_sys.ad_categories["male"]["child"] and not ad_sys.ad_categories["general"]:
                logger.warning("No ads in 'male/child' or 'general' to assert targeted play.")
            else:
                # Check if it stabilized to male_child, or fell back to general/default if male_child ads don't exist
                expected_stable = "male_child"
                if not ad_sys.ad_categories["male"]["child"]:
                    expected_stable = "general" if ad_sys.ad_categories["general"] else "default"
                
                assert ad_sys.last_stable_category_voted == expected_stable, \
                       f"Expected stable category {expected_stable}, got {ad_sys.last_stable_category_voted}"

    except AssertionError as e:
        logger.error(f"Self-test assertion failed: {e}")
    except Exception as e:
        logger.exception("Error during AdSystem self-test:")
    finally:
        if ad_sys:
            ad_sys.cleanup()
        logger.info("AdSystem self-test finished.")