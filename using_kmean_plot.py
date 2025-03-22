import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO
import supervision as sv
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics.nn.tasks import DetectionModel  # Custom DetectionModel class
from sklearn.cluster import KMeans

# =============================================================================
# Global Parameters & Paths
# =============================================================================
VIDEO_PATH = r"Videos/Video8.mp4"       # Video for detection and drawing
BACKGROUND_IMG_PATH = "input1.png"        # Background image (canvas)
OUTPUT_VIDEO_PATH = "final_output.mp4"    # Output video file
DESIRED_CLASS = 0                         # Player class index

YOLO_CONFIDENCE = 0.3
YOLO_IOU = 0.5

DOT_PARAMS = {'radius': 3, 'color': (0, 0, 0), 'thickness': -1}  # Dot properties

# =============================================================================
# Functional Definitions
# =============================================================================

def load_model(model_path, conf=YOLO_CONFIDENCE, iou=YOLO_IOU):
    torch.serialization.add_safe_globals([DetectionModel])
    model = YOLO(model_path)
    model.conf = conf
    model.iou = iou
    return model

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise Exception(f"Could not load image at {path}")
    return img

def create_tracker():
    return DeepSort(
        max_age=30,
        n_init=3,
        nms_max_overlap=0.7,
        max_cosine_distance=0.2,
        nn_budget=100,
        embedder="mobilenet",
        half=True,
        bgr=True,
        embedder_gpu=True
    )

def get_polygon_from_user(image, title, num_points):
    plt.figure(figsize=(10,6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    pts = []
    for _ in range(num_points):
        pt = plt.ginput(1, timeout=0)[0]
        pts.append(pt)
        plt.scatter(pt[0], pt[1], color='red', marker='o')
        if len(pts) > 1:
            plt.plot([pts[-2][0], pts[-1][0]], [pts[-2][1], pts[-1][1]], 'r-', lw=2)
    if num_points >= 3:
        plt.plot([pts[-1][0], pts[0][0]], [pts[-1][1], pts[0][1]], 'r-', lw=2)
    plt.title("Close window when done.")
    plt.show()
    return np.array(pts, dtype=np.float32)

def get_points_from_user(image, title, num_points):
    """Gets a set of points without connecting lines."""
    plt.figure(figsize=(10,6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    pts = []
    for _ in range(num_points):
        pt = plt.ginput(1, timeout=0)[0]
        pts.append(pt)
        plt.scatter(pt[0], pt[1], color='magenta', marker='o')
    plt.title("Close window when done.")
    plt.show()
    return np.array(pts, dtype=np.float32)

def get_two_points_from_user(image, title):
    return get_points_from_user(image, title, 2)

def compute_homography(src_pts, dst_pts):
    H, _ = cv2.findHomography(src_pts, dst_pts)
    return H

def transform_points(points, H):
    pts = cv2.perspectiveTransform(points.reshape(-1, 1, 2), H)
    return pts.reshape(-1, 2).astype(int)

def filter_detections_by_class(detections, desired_class):
    filtered_boxes = []
    filtered_conf = []
    filtered_cls = []
    for i, box in enumerate(detections.xyxy):
        if int(detections.class_id[i]) != desired_class:
            continue
        filtered_boxes.append(box)
        filtered_conf.append(detections.confidence[i])
        filtered_cls.append(detections.class_id[i])
    if len(filtered_boxes) > 0:
        detections.xyxy = np.array(filtered_boxes)
        detections.confidence = np.array(filtered_conf)
        detections.class_id = np.array(filtered_cls)
    else:
        detections = sv.Detections.empty()
    return detections

def update_tracker_with_detections(detections, tracker, frame):
    results_list = []
    if len(detections) > 0:
        for i, box in enumerate(detections.xyxy):
            xmin, ymin, xmax, ymax = map(int, box)
            w, h = xmax - xmin, ymax - ymin
            confidence = float(detections.confidence[i])
            class_id = int(detections.class_id[i])
            if w > 20 and h > 20:
                results_list.append([[xmin, ymin, w, h], confidence, class_id])
        tracks = tracker.update_tracks(results_list, frame=frame)
        active_tracks = [t for t in tracks if t.is_confirmed() and t.time_since_update < 5]
    else:
        tracker.update_tracks([], frame=frame)
        active_tracks = []
    return active_tracks

def assign_teams_kmeans_points(active_tracks, video_polygon, initial_centers):
    """
    Collects bottom-center points from active tracks that fall within the video court.
    Then runs k-means clustering (with K=2) on these points using the provided initial centers.
    Returns a dictionary mapping track_id to team label.
    """
    points = []
    track_indices = []
    for idx, t in enumerate(active_tracks):
        box = t.to_tlbr()
        bottom_x = (box[0] + box[2]) / 2
        bottom_y = box[3]
        if cv2.pointPolygonTest(video_polygon, (bottom_x, bottom_y), False) >= 0:
            points.append([bottom_x, bottom_y])
            track_indices.append(idx)
    team_labels = {}
    if len(points) >= 2:
        Z = np.float32(points)
        kmeans = KMeans(n_clusters=2, init=initial_centers, n_init=1, random_state=42)
        kmeans.fit(Z)
        labels = kmeans.labels_
        for i, idx in enumerate(track_indices):
            team_labels[active_tracks[idx].track_id] = "Team A" if labels[i] == 0 else "Team B"
    return team_labels

def draw_canvas(background_img, background_polygon, video_polygon, active_tracks, H, dot_params):
    canvas = background_img.copy()
    bg_poly = background_polygon.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(canvas, [bg_poly], isClosed=True, color=(0,255,0), thickness=2)
    for t in active_tracks:
        box = t.to_tlbr()
        bottom_x = (box[0] + box[2]) / 2
        bottom_y = box[3]
        if cv2.pointPolygonTest(video_polygon, (bottom_x, bottom_y), False) >= 0:
            pt_video = np.array([[bottom_x, bottom_y]], dtype=np.float32)
            pt_canvas = transform_points(pt_video, H)
            cv2.circle(canvas, (int(pt_canvas[0][0]), int(pt_canvas[0][1])), dot_params['radius'], dot_params['color'], dot_params['thickness'])
    return canvas

def annotate_frame(frame, tracked_detections, team_labels):
    annotated = frame.copy()
    annotated = sv.BoxAnnotator(thickness=2, color=sv.Color.from_hex('#00FF00')).annotate(scene=annotated, detections=tracked_detections)
    for det in tracked_detections.xyxy:
        bottom_x = int((det[0] + det[2]) / 2)
        bottom_y = int(det[3])
        idx_arr = np.where((tracked_detections.xyxy == det).all(axis=1))[0]
        if len(idx_arr) > 0:
            track_id = tracked_detections.tracker_id[idx_arr[0]]
            team = team_labels.get(track_id, "Team ?")
            cv2.putText(annotated, f"{team} ID:{track_id}", (bottom_x, bottom_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    return annotated

def process_video(video_path, background_img, background_polygon, video_polygon, H, initial_centers, desired_class, tracker, output_video_path, team_assignment_func):
    cap = cv2.VideoCapture(video_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps_video, (width, height))
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    
    dot_params = DOT_PARAMS.copy()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_copy = frame.copy()
        result = PLAYER_DETECTION_MODEL.predict(frame, conf=0.1)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = filter_detections_by_class(detections, desired_class)
        
        active_tracks = update_tracker_with_detections(detections, tracker, frame)
        
        team_labels = team_assignment_func(active_tracks, video_polygon, initial_centers)
        
        canvas_current = draw_canvas(background_img, background_polygon, video_polygon, active_tracks, H, dot_params)
        
        output_frame = frame_copy.copy()
        canvas_h, canvas_w = canvas_current.shape[:2]
        if output_frame.shape[0] >= 20 + canvas_h and output_frame.shape[1] >= 20 + canvas_w:
            output_frame[20:20+canvas_h, 20:20+canvas_w] = canvas_current
        
        if active_tracks:
            tracked_detections = sv.Detections(
                xyxy=np.array([t.to_tlbr() for t in active_tracks]),
                confidence=np.array([t.det_conf for t in active_tracks]),
                class_id=np.array([t.det_class for t in active_tracks]),
                tracker_id=np.array([t.track_id for t in active_tracks])
            )
        else:
            tracked_detections = sv.Detections.empty()
        
        annotated_frame = annotate_frame(output_frame, tracked_detections, team_labels)
        cv2.imshow("Output", annotated_frame)
        out_writer.write(annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()

# =============================================================================
# Main Execution
# =============================================================================

# Load the model.
PLAYER_DETECTION_MODEL = load_model(r"Models/volleyPlayers.pt", YOLO_CONFIDENCE, YOLO_IOU)

# Step 2: Draw court on video's first frame.
cap = cv2.VideoCapture(VIDEO_PATH)
ret, first_frame = cap.read()
if not ret:
    raise Exception("Could not read first frame from video.")
first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
cap.release()
video_polygon = get_polygon_from_user(first_frame, "VIDEO: Draw Court - Click 4 Points", 4)
_ = get_two_points_from_user(first_frame, "VIDEO: Draw Mid-Line - Click 2 Points")  # Mid-line not used further.

# Step 3: Draw 6 points for clustering on video’s first frame.
# First 3 points for Team A; next 3 for Team B.
print("Draw 6 points for clustering (first 3 for Team A, next 3 for Team B)")
cluster_points = get_points_from_user(first_frame, "VIDEO: Draw 6 Points for Clustering", 6)
# Compute initial centers:
initial_center_A = np.mean(cluster_points[:3], axis=0)
initial_center_B = np.mean(cluster_points[3:], axis=0)
initial_centers = np.array([initial_center_A, initial_center_B], dtype=np.float32)

# Step 4: Prepare background canvas. Let user draw court on background image.
background_img = load_image(BACKGROUND_IMG_PATH)
bg_polygon_draw = get_polygon_from_user(background_img, "BACKGROUND: Draw Court - Click 4 Points", 4)
background_polygon = bg_polygon_draw

# Step 5: Compute homography.
H = compute_homography(video_polygon, background_polygon)

# Create tracker.
tracker = create_tracker()

# Process video using k-means based team assignment using the manually defined initial centers.
process_video(VIDEO_PATH, background_img, background_polygon, video_polygon, H, initial_centers, DESIRED_CLASS, tracker, OUTPUT_VIDEO_PATH, assign_teams_kmeans_points)
