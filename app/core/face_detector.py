# face_detector.py

import time
import cv2
import numpy as np
import logging
from math import ceil

import config # Main project config

logger = logging.getLogger(__name__)

# --- ULFGFD Helper Functions ---
def _ulfd_generate_priors(image_size_wh_list, shrinkage_list_of_strides, min_boxes_per_feature_map):
    image_width, image_height = image_size_wh_list
    priors = []
    for index in range(len(shrinkage_list_of_strides)):
        stride_val = shrinkage_list_of_strides[index]
        f_w = int(ceil(image_width / stride_val))
        f_h = int(ceil(image_height / stride_val))
        for j in range(f_h):
            for i in range(f_w):
                x_center = (i + 0.5) / f_w
                y_center = (j + 0.5) / f_h
                for min_box_pixel_size in min_boxes_per_feature_map[index]:
                    w_norm = min_box_pixel_size / image_width
                    h_norm = min_box_pixel_size / image_height
                    priors.append([x_center, y_center, w_norm, h_norm])
    priors_np = np.array(priors, dtype=np.float32)
    logger.info(f"ULGFD: Generated {priors_np.shape[0]} priors.")
    return np.clip(priors_np, 0.0, 1.0)

def _ulfd_convert_locations_to_boxes(locations, priors, center_variance, size_variance):
    if len(priors.shape) + 1 == len(locations.shape):
        priors = np.expand_dims(priors, 0)
    return np.concatenate([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        np.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], axis=len(locations.shape) - 1)

def _ulfd_center_form_to_corner_form(locations_center_xywh):
    return np.concatenate([
        locations_center_xywh[..., :2] - locations_center_xywh[..., 2:] / 2,
        locations_center_xywh[..., :2] + locations_center_xywh[..., 2:] / 2
    ], axis=len(locations_center_xywh.shape) - 1)

def _ulfd_predict_postprocess(img_width, img_height, confidences_batch, boxes_batch,
                             prob_threshold, iou_threshold):
    class_index = 1 
    probs_for_face_class = confidences_batch[:, class_index]
    mask = probs_for_face_class > prob_threshold
    if not np.any(mask): return np.array([]), np.array([])
    probs_masked = probs_for_face_class[mask]
    boxes_masked = boxes_batch[mask, :]
    box_probs_for_nms = np.concatenate([boxes_masked, probs_masked.reshape(-1, 1)], axis=1)
    rects_for_cv_nms, scores_for_cv_nms = [], []
    for i in range(box_probs_for_nms.shape[0]):
        x1_n, y1_n, x2_n, y2_n, score = box_probs_for_nms[i]
        rects_for_cv_nms.append([x1_n, y1_n, x2_n - x1_n, y2_n - y1_n]) # x, y, w, h (normalized)
        scores_for_cv_nms.append(score)
    if not rects_for_cv_nms: return np.array([]), np.array([])
    
    indices = cv2.dnn.NMSBoxes(rects_for_cv_nms, scores_for_cv_nms, prob_threshold, iou_threshold)
    
    final_boxes_scaled, final_probs = [], []
    if indices is not None and len(indices) > 0: # Check if indices is not None
        if isinstance(indices, tuple): indices = indices[0]
        if indices.ndim > 1: indices = indices.flatten()
        for i in indices:
            norm_x1, norm_y1, norm_w, norm_h = rects_for_cv_nms[i]
            score = scores_for_cv_nms[i]
            x1 = int(norm_x1 * img_width)
            y1 = int(norm_y1 * img_height)
            x2 = int((norm_x1 + norm_w) * img_width)
            y2 = int((norm_y1 + norm_h) * img_height)
            final_boxes_scaled.append([x1, y1, x2, y2])
            final_probs.append(score)
    return np.array(final_boxes_scaled, dtype=np.int32), np.array(final_probs, dtype=np.float32)

def _expand_box(box_xyxy, img_width, img_height, expansion_factor):
    x1, y1, x2, y2 = box_xyxy
    box_w, box_h = x2 - x1, y2 - y1
    expand_w, expand_h = int(box_w * expansion_factor), int(box_h * expansion_factor)
    ex1, ey1 = max(0, x1 - expand_w), max(0, y1 - expand_h)
    ex2, ey2 = min(img_width - 1, x2 + expand_w), min(img_height - 1, y2 + expand_h)
    return [ex1, ey1, ex2, ey2]
# --- END Helper Functions ---

class FaceDetector:
    def __init__(self):
        logger.info("Initializing FaceDetector (ULGFD faces, CV-DNN Age/Gender)...")
        
        self.ulfd_input_size_wh = tuple(config.ULFD_INPUT_SIZE_WH)
        self.ulfd_preproc_scalefactor = config.ULFD_PREPROC_SCALEFACTOR
        self.ulfd_preproc_mean = config.ULFD_PREPROC_MEAN
        self.ulfd_conf_threshold = config.ULFD_CONF_THRESHOLD
        self.ulfd_iou_threshold = config.ULFD_IOU_THRESHOLD
        self.ag_crop_expansion_factor = config.AG_CROP_EXPANSION_FACTOR
        self.priors = _ulfd_generate_priors(
            config.ULFD_INPUT_SIZE_WH, config.ULFD_PRIOR_STRIDES, config.ULFD_PRIOR_MIN_BOXES)
        self.ulfd_center_variance = config.ULFD_PRIOR_CENTER_VARIANCE
        self.ulfd_size_variance = config.ULFD_PRIOR_SIZE_VARIANCE

        self.ag_classifier_input_size_wh = tuple(config.AG_CLASSIFIER_INPUT_SIZE_WH)
        self.ag_classifier_mean_vals = config.AG_CLASSIFIER_MEAN_VALS
        self.gender_list = config.GENDER_LIST_CV_DNN
        self.age_intervals = config.AGE_INTERVALS_CV_DNN

        self.ulfd_net = None
        self.age_net_cv_dnn = None
        self.gender_net_cv_dnn = None
        self._load_models()

        self.frame_count = 0
        self.total_face_inference_time = 0
        self.total_ag_inference_time = 0
        logger.info(f"FaceDetector (ULGFD + CV-DNN A/G) initialized.")

    def _load_models(self):
        logger.info("Loading ONNX models via OpenCV DNN...")
        try:
            self.ulfd_net = cv2.dnn.readNetFromONNX(config.ULFD_ONNX_MODEL_PATH)
            logger.info(f"ULGFD Face Model loaded: {config.ULFD_ONNX_MODEL_PATH}")
            self.gender_net_cv_dnn = cv2.dnn.readNetFromONNX(config.GENDER_MODEL_PATH_CV_DNN)
            logger.info(f"Gender Model loaded: {config.GENDER_MODEL_PATH_CV_DNN}")
            self.age_net_cv_dnn = cv2.dnn.readNetFromONNX(config.AGE_MODEL_PATH_CV_DNN)
            logger.info(f"Age Model loaded: {config.AGE_MODEL_PATH_CV_DNN}")
            logger.info(f"All models loaded successfully using OpenCV DNN.")
        except Exception as e:
            logger.exception(f"Error loading models: {e}")
            raise

    def _preprocess_for_detector(self, frame: np.ndarray) -> np.ndarray:
        return cv2.dnn.blobFromImage(frame,
                                     scalefactor=self.ulfd_preproc_scalefactor,
                                     size=self.ulfd_input_size_wh,
                                     mean=self.ulfd_preproc_mean,
                                     swapRB=False, crop=False) 

    def _preprocess_for_classifiers(self, face_regions: list) -> np.ndarray:
        if not face_regions: return None
        valid_regions = [fr for fr in face_regions if fr is not None and fr.size > 0]
        if not valid_regions: return None
        try:
            return cv2.dnn.blobFromImages(valid_regions,
                                          scalefactor=1.0,
                                          size=self.ag_classifier_input_size_wh,
                                          mean=self.ag_classifier_mean_vals,
                                          swapRB=False, crop=False)
        except cv2.error as e:
            logger.error(f"Error creating blob for A/G classifiers: {e}. Regions: {len(valid_regions)}")
            return None

    def detect_faces(self, processed_detector_blob: np.ndarray):
        start_time = time.perf_counter()
        self.ulfd_net.setInput(processed_detector_blob)
        try:
            # Attempt to get output layers by common names, fallback to all ordered outputs
            output_names = self.ulfd_net.getUnconnectedOutLayersNames()
            if "scores" in output_names and "boxes" in output_names:
                 outputs = self.ulfd_net.forward(["scores", "boxes"])
            else: # Fallback for models not using these exact names
                outputs = self.ulfd_net.forward(output_names) 
                if len(outputs) < 2:
                    logger.error(f"ULGFD: Not enough outputs. Got {len(outputs)}, Names: {output_names}")
                    return [],[],0.0
                logger.debug(f"ULGFD outputs by order. Names: {output_names}")
        except cv2.error as e:
            logger.error(f"ULGFD forward pass error: {e}")
            return [], [], 0.0

        confidences_raw_batch, locations_raw_batch = outputs[0], outputs[1]
        inference_time_ms = (time.perf_counter() - start_time) * 1000
        conf_single, loc_single = confidences_raw_batch[0], locations_raw_batch[0]
        
        decoded_boxes_center_norm = _ulfd_convert_locations_to_boxes(
            loc_single, self.priors, self.ulfd_center_variance, self.ulfd_size_variance)
        decoded_boxes_corner_norm = _ulfd_center_form_to_corner_form(decoded_boxes_center_norm)
        
        net_w, net_h = self.ulfd_input_size_wh
        final_boxes, final_probs = _ulfd_predict_postprocess(
            net_w, net_h, conf_single, decoded_boxes_corner_norm,
            self.ulfd_conf_threshold, self.ulfd_iou_threshold)
        return final_boxes.tolist(), final_probs.tolist(), inference_time_ms

    def classify_age_gender(self, face_regions_blob: np.ndarray):
        if face_regions_blob is None or face_regions_blob.shape[0] == 0:
            return [], [], 0.0

        num_faces = face_regions_blob.shape[0]
        genders, ages = ["U"] * num_faces, ["U"] * num_faces
        start_time = time.perf_counter()
        try:
            self.gender_net_cv_dnn.setInput(face_regions_blob)
            gender_out = self.gender_net_cv_dnn.forward()
            for i in range(num_faces): genders[i] = self.gender_list[np.argmax(gender_out[i])]
        except Exception as e: logger.error(f"Gender classification error: {e}")
        try:
            self.age_net_cv_dnn.setInput(face_regions_blob)
            age_out = self.age_net_cv_dnn.forward()
            for i in range(num_faces): ages[i] = self.age_intervals[np.argmax(age_out[i])]
        except Exception as e: logger.error(f"Age classification error: {e}")
        return genders, ages, (time.perf_counter() - start_time) * 1000

    def process_frame(self, original_frame: np.ndarray):
        self.frame_count += 1
        total_inference_this_frame_ms = 0
        
        detector_blob = self._preprocess_for_detector(original_frame)
        boxes_on_net_input, confs_on_net_input, face_inf_time = self.detect_faces(detector_blob)
        self.total_face_inference_time += face_inf_time
        total_inference_this_frame_ms += face_inf_time
        
        faces_info_list = []
        if not boxes_on_net_input:
            return faces_info_list, total_inference_this_frame_ms

        orig_h, orig_w = original_frame.shape[:2]
        net_w, net_h = self.ulfd_input_size_wh
        scale_x, scale_y = orig_w / net_w, orig_h / net_h

        # Prepare inputs for A/G classification
        # Store original detected boxes and the regions to be classified
        # This list will hold tuples: (original_scaled_box, region_for_classifier OR None)
        classification_inputs = [] 

        for i, box_ni in enumerate(boxes_on_net_input):
            x1_o = int(box_ni[0] * scale_x); y1_o = int(box_ni[1] * scale_y)
            x2_o = int(box_ni[2] * scale_x); y2_o = int(box_ni[3] * scale_y)
            x1_o,y1_o = max(0,x1_o), max(0,y1_o)
            x2_o,y2_o = min(orig_w-1,x2_o), min(orig_h-1,y2_o)
            
            original_scaled_box = [x1_o, y1_o, x2_o, y2_o]

            if x2_o <= x1_o or y2_o <= y1_o:
                classification_inputs.append({'orig_box': original_scaled_box, 'crop': None, 'conf': confs_on_net_input[i]})
                continue

            expanded_box = _expand_box(original_scaled_box, orig_w, orig_h, self.ag_crop_expansion_factor)
            ex1, ey1, ex2, ey2 = expanded_box
            
            if ex2 <= ex1 or ey2 <= ey1:
                classification_inputs.append({'orig_box': original_scaled_box, 'crop': None, 'conf': confs_on_net_input[i]})
            else:
                crop = original_frame[ey1:ey2, ex1:ex2]
                classification_inputs.append({'orig_box': original_scaled_box, 'crop': crop, 'conf': confs_on_net_input[i]})

        # Batch preprocess valid crops for classifiers
        valid_crops_for_ag = [item['crop'] for item in classification_inputs if item['crop'] is not None and item['crop'].size > 0]
        
        genders_classified, ages_classified = [], []
        ag_inf_time = 0
        if valid_crops_for_ag:
            classifier_blob = self._preprocess_for_classifiers(valid_crops_for_ag)
            if classifier_blob is not None:
                genders_classified, ages_classified, ag_inf_time = self.classify_age_gender(classifier_blob)
        
        self.total_ag_inference_time += ag_inf_time
        total_inference_this_frame_ms += ag_inf_time

        # Collate final results
        crop_classified_idx = 0
        for item in classification_inputs:
            gender, age = "U", "U"
            if item['crop'] is not None and item['crop'].size > 0:
                if crop_classified_idx < len(genders_classified): # Check bounds
                    gender = genders_classified[crop_classified_idx]
                    age = ages_classified[crop_classified_idx]
                    crop_classified_idx += 1
            
            faces_info_list.append({
                'box_original': item['orig_box'],
                'gender': gender, 'age': age,
                'confidence': float(item['conf'])
            })
            
        return faces_info_list, total_inference_this_frame_ms

    def get_average_inference_times(self):
        if self.frame_count == 0: return 0,0
        avg_face = self.total_face_inference_time / self.frame_count
        avg_ag = self.total_ag_inference_time / self.frame_count
        return avg_face, avg_ag

# --- Self-Test (can be adapted from previous version) ---
if __name__ == '__main__':
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
    logger.info("Running FaceDetector self-test (ULGFD faces, CV-DNN Age/Gender)...")
    # ... (self-test code, similar to the last working one for face_detector.py) ...
    dummy_frame_orig = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(dummy_frame_orig, (250, 180), (390, 330), (70, 70, 70), -1)
    cv2.circle(dummy_frame_orig, (300, 230), 10, (20,20,20), -1)
    cv2.circle(dummy_frame_orig, (340, 230), 10, (20,20,20), -1)
    cv2.line(dummy_frame_orig, (300, 280), (340,280), (20,20,20), 2)
    try:
        detector = FaceDetector()
        logger.info("Processing dummy frame...")
        num_test_frames = 10; faces = []
        for i in range(num_test_frames + 1):
            faces, total_time_ms = detector.process_frame(dummy_frame_orig.copy())
            if i == 0: logger.info(f"Initial frame total time: {total_time_ms:.2f} ms")
        if faces:
            logger.info(f"Detected {len(faces)} face(s) in last test frame:")
            for i, face_info in enumerate(faces): logger.info(f"  Face {i+1}: {face_info}")
        else: logger.info("No faces detected in dummy frame.")
        avg_face, avg_ag = detector.get_average_inference_times()
        logger.info(f"Avg FaceDetTime: {avg_face:.2f}ms, Avg AGClassTime: {avg_ag:.2f}ms")
    except Exception as e: logger.exception("Self-test error:")
    logger.info("FaceDetector self-test finished.")