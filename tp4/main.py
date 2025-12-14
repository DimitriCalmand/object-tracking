import os
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2
from tqdm import tqdm
import random
from KalmanFilter import KalmanFilter
import onnxruntime as ort

class ReIDExtractor:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.roi_width = 64
        self.roi_height = 128
        self.roi_means = np.array([0.485, 0.456, 0.406]) * 255
        self.roi_stds = np.array([0.229, 0.224, 0.225]) * 255
        
    def preprocess_patch(self, im_crops):
        roi_input = cv2.resize(im_crops, (self.roi_width, self.roi_height))
        roi_input = cv2.cvtColor(roi_input, cv2.COLOR_BGR2RGB)
        roi_input = (np.asarray(roi_input).astype(np.float32) - self.roi_means) / self.roi_stds
        roi_input = np.moveaxis(roi_input, -1, 0)
        object_patch = roi_input.astype('float32')
        return object_patch
    
    def extract_features(self, image, bbox):
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        patch = image[y:y+h, x:x+w]
        if patch.size == 0:
            return None
            
        preprocessed = self.preprocess_patch(patch)
        preprocessed = np.expand_dims(preprocessed, axis=0)
        
        input_name = self.session.get_inputs()[0].name
        features = self.session.run(None, {input_name: preprocessed})[0]
        features = features.flatten()
        features = features / np.linalg.norm(features)
        
        return features

class Track:
    def __init__(self, id, position, feature=None, missed=0):
        self.id = id
        self.position = position
        self.missed = missed
        self.feature = feature
        
        dt = 1
        u_x, u_y = 0, 0
        std_acc = 1
        x_std_meas = 0.1
        y_std_meas = 0.1
        
        self.kf = KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
        
        x, y, w, h = position
        cx = x + w / 2
        cy = y + h / 2
        self.kf.x_k = np.array([[cx], [cy], [0], [0]])
        
    def predict(self):
        self.kf.predict()
        cx = self.kf.x_k[0][0]
        cy = self.kf.x_k[1][0]
        
        _, _, w, h = self.position
        
        x = cx - w / 2
        y = cy - h / 2
        self.position = (x, y, w, h)
        
    def update(self, detection_position, feature=None):
        x, y, w, h = detection_position
        cx = x + w / 2
        cy = y + h / 2
        
        z_k = np.array([[cx], [cy]])
        self.kf.update(z_k)
        
        updated_cx = self.kf.x_k[0][0]
        updated_cy = self.kf.x_k[1][0]
        
        new_x = updated_cx - w / 2
        new_y = updated_cy - h / 2
        self.position = (new_x, new_y, w, h)
        
        if feature is not None:
            self.feature = feature

def load_det(det_path, separator=","):
    with open(det_path, 'r') as f:
        lines = f.readlines()
    detections = {}
    list_order_label = ["frame","id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"]
    for line in lines:
        
        elem = line.split(separator)
        frame = str(elem[0])
        if detections.get(frame) is None:
            detections[frame] = []

        bounding_box = {}
        for i in range(1, len(list_order_label)):
            bounding_box[list_order_label[i]] = float(elem[i])
        detections[frame].append(bounding_box)
    return detections

def get_position(detection : dict):
    x = detection["bb_left"]
    y = detection["bb_top"]
    w = detection["bb_width"]
    h = detection["bb_height"]

    return (x, y, w, h)

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0
    
    return inter_area / union_area

def compute_feature_similarity(feat1, feat2):
    if feat1 is None or feat2 is None:
        return 0.0
    
    euclidean_dist = np.linalg.norm(feat1 - feat2)
    normalized_similarity = 1 / (1 + euclidean_dist)
    
    return normalized_similarity

def create_similarity_matrix(tracks, det_positions, det_features, alpha=0.5, beta=0.5):
    num_tracks = len(tracks)
    num_dets = len(det_positions)
    
    similarity_matrix = np.zeros((num_tracks, num_dets))

    for t in range(num_tracks):
        for d in range(num_dets):
            iou_score = compute_iou(tracks[t].position, det_positions[d])
            
            feature_sim = compute_feature_similarity(tracks[t].feature, det_features[d])
            
            combined_score = alpha * iou_score + beta * feature_sim
            
            similarity_matrix[t, d] = combined_score
            
    return similarity_matrix

reid_extractor = ReIDExtractor("reid_osnet_x025_market1501.onnx")

detections_dict = load_det("../tp2/ADL-Rundle-6/det/Yolov5l/det.txt", separator=" ")

frames = sorted([int(f) for f in detections_dict.keys()])

active_tracks = []
next_id = 1
max_missed_frames = 10
iou_threshold = 0.3
alpha = 0.5
beta = 0.5

tracking_results = {}

image_dir = "../tp2/ADL-Rundle-6/img1/"

for frame in tqdm(frames):
    dets = detections_dict[str(frame)]
    det_positions = [get_position(d) for d in dets]
    
    img_path = os.path.join(image_dir, f"{frame:06d}.jpg")
    img = cv2.imread(img_path)
    
    det_features = []
    for det_pos in det_positions:
        if img is not None:
            feature = reid_extractor.extract_features(img, det_pos)
            det_features.append(feature)
        else:
            det_features.append(None)
    
    if active_tracks:
        for track in active_tracks:
            track.predict()
            
        sim_matrix = create_similarity_matrix(active_tracks, det_positions, det_features, alpha, beta)
        
        cost_matrix = 1 - sim_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches = []
        matched_tracks = set()
        matched_dets = set()
        
        for r, c in zip(row_ind, col_ind):
            if sim_matrix[r, c] >= iou_threshold:
                matches.append((r, c))
                matched_tracks.add(r)
                matched_dets.add(c)
        
        for track_idx, det_idx in matches:
            active_tracks[track_idx].update(det_positions[det_idx], det_features[det_idx])
            active_tracks[track_idx].missed = 0
        
        new_active_tracks = []
        for i, track in enumerate(active_tracks):
            if i not in matched_tracks:
                track.missed += 1
                if track.missed <= max_missed_frames:
                    new_active_tracks.append(track)
            else:
                new_active_tracks.append(track)
        active_tracks = new_active_tracks
        
        for i, det_pos in enumerate(det_positions):
            if i not in matched_dets:
                new_track = Track(next_id, det_pos, det_features[i])
                active_tracks.append(new_track)
                next_id += 1
    else:
        for i, det_pos in enumerate(det_positions):
            new_track = Track(next_id, det_pos, det_features[i])
            active_tracks.append(new_track)
            next_id += 1
    
    tracking_results[frame] = [(t.id, t.position) for t in active_tracks]
    

print("Tracking completed.")

output_txt = "ADL-Rundle-6.txt"
with open(output_txt, 'w') as f:
    for frame in frames:
        for track_id, position in tracking_results[frame]:
            x, y, w, h = position
            line = f"{frame},{track_id},{int(x)},{int(y)},{int(w)},{int(h)},1,-1,-1,-1\n"
            f.write(line)

print(f"Tracking results saved to {output_txt}")

print("Processing visualization...")

output_video = "tracking_output.avi"

sample_img = cv2.imread(os.path.join(image_dir, "000001.jpg"))
if sample_img is None:
    print("Could not load sample image. Check path.")
else:
    height, width, _ = sample_img.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, 30.0, (width, height))

    track_colors = {}

    def get_color(track_id):
        if track_id not in track_colors:
            track_colors[track_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        return track_colors[track_id]

    for frame in tqdm(frames):
        img_path = os.path.join(image_dir, f"{frame:06d}.jpg")
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Could not load image for frame {frame}")
            continue
        
        for track_id, position in tracking_results[frame]:
            x, y, w, h = position
            color = get_color(track_id)
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
            cv2.putText(img, str(track_id), (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        out.write(img)

    out.release()
    print(f"Tracking video saved as {output_video}")
