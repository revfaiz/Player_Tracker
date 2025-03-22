# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import torch
# from ultralytics import YOLO
# import supervision as sv
# from deep_sort_realtime.deepsort_tracker import DeepSort
# from ultralytics.nn.tasks import DetectionModel  # Custom DetectionModel class

# # --- Setup Environment & Model ---
# torch.serialization.add_safe_globals([DetectionModel])
# PLAYER_DETECTION_MODEL = YOLO(r"Models\UpdatedPlayer.pt")
# # Set YOLO thresholds (adjust as desired)
# PLAYER_DETECTION_MODEL.conf = 0.3  
# PLAYER_DETECTION_MODEL.iou  = 0.5  

# # --- Input Paths ---
# VIDEO_PATH = r"Videos/Video8.mp4"       # Video for detection and court drawing (first frame)
# BACKGROUND_IMG_PATH = "input1.png"        # Background image (entire image is the canvas)
# OUTPUT_VIDEO_PATH = "final_output.mp4"    # Final output video

# # === STEP 1: Draw Court on Video's First Frame ===
# cap = cv2.VideoCapture(VIDEO_PATH)
# ret, first_frame = cap.read()
# if not ret:
#     raise Exception("Could not read first frame from video.")
# first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
# cap.release()

# plt.figure(figsize=(10, 6))
# plt.imshow(first_frame_rgb)
# plt.title("VIDEO: Draw Court - Click 4 Points")
# video_polygon = []
# for _ in range(4):
#     pt = plt.ginput(1, timeout=0)[0]
#     video_polygon.append(pt)
#     plt.scatter(pt[0], pt[1], color='red', marker='o')
#     if len(video_polygon) > 1:
#         plt.plot([video_polygon[-2][0], video_polygon[-1][0]],
#                  [video_polygon[-2][1], video_polygon[-1][1]], 'r-', lw=2)
# # Close the polygon by connecting last to first
# plt.plot([video_polygon[-1][0], video_polygon[0][0]],
#          [video_polygon[-1][1], video_polygon[0][1]], 'r-', lw=2)
# video_polygon = np.array(video_polygon, dtype=np.float32)
# plt.title("Close window when done.")
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.imshow(first_frame_rgb)
# plt.title("VIDEO: Draw Mid-Line - Click 2 Points")
# video_halfline = np.array(plt.ginput(2, timeout=0), dtype=np.float32)
# plt.scatter(video_halfline[:, 0], video_halfline[:, 1], color='blue', marker='x')
# plt.plot(video_halfline[:, 0], video_halfline[:, 1], 'b-', lw=2)
# plt.title("Close window when done.")
# plt.show()

# # === STEP 2: Prepare Background Canvas Automatically ===
# background_img = cv2.imread(BACKGROUND_IMG_PATH)
# if background_img is None:
#     raise Exception("Could not load background image.")
# # Use the entire background image as canvas.
# bg_h, bg_w = background_img.shape[:2]
# # Define background polygon as the entire image.
# background_polygon = np.array([[0, 0], [bg_w, 0], [bg_w, bg_h], [0, bg_h]], dtype=np.float32)

# # === STEP 3: Compute Homography from Video Court to Background Canvas ===
# # This homography maps points from the video-drawn court (video_polygon)
# # to the background canvas (which is the whole background image).
# H, _ = cv2.findHomography(video_polygon, background_polygon)
# # (Optional: you may transform the mid-line if needed.)

# # === STEP 4: Prepare for Tracking & Dot Drawing ===
# # Dot parameters for marking players on the canvas
# DOT_RADIUS = 3
# DOT_COLOR = (0, 0, 0)  # Black
# DOT_THICKNESS = -1     # Filled circle

# # Initialize DeepSORT tracker to get unique IDs.
# tracker = DeepSort(
#     max_age=30,
#     n_init=3,
#     nms_max_overlap=0.7,
#     max_cosine_distance=0.2,
#     nn_budget=100,
#     embedder="mobilenet",
#     half=True,
#     bgr=True,
#     embedder_gpu=True
# )

# def transform_points(points, H):
#     pts = cv2.perspectiveTransform(points.reshape(-1, 1, 2), H)
#     return pts.reshape(-1, 2).astype(int)

# # === STEP 5: Process Video Frames, Detect & Track, and Overlay Canvas ===
# cap = cv2.VideoCapture(VIDEO_PATH)
# fps_video = cap.get(cv2.CAP_PROP_FPS)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# out_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps_video, (width, height))

# cv2.namedWindow("Output", cv2.WINDOW_NORMAL)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame_copy = frame.copy()
    
#     # Run YOLO detection on the frame
#     result = PLAYER_DETECTION_MODEL.predict(frame, conf=0.1)[0]
#     detections = sv.Detections.from_ultralytics(result)
    
#     # Update tracker with detections for unique IDs (using raw detections here)
#     if len(detections) > 0:
#         results_list = []
#         for i, box in enumerate(detections.xyxy):
#             xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
#             w, h = xmax - xmin, ymax - ymin
#             confidence = float(detections.confidence[i])
#             class_id = int(detections.class_id[i])
#             if w > 20 and h > 20:
#                 results_list.append([[xmin, ymin, w, h], confidence, class_id])
#         tracks = tracker.update_tracks(results_list, frame=frame)
#         active_tracks = [t for t in tracks if t.is_confirmed() and t.time_since_update < 5]
#     else:
#         tracker.update_tracks([], frame=frame)
#         active_tracks = []
    
#     # Create a copy of the background canvas (use the background image as is)
#     canvas_current = background_img.copy()
    
#     # (Optionally) Draw the background court boundary
#     bg_poly = background_polygon.reshape((-1, 1, 2)).astype(np.int32)
#     cv2.polylines(canvas_current, [bg_poly], isClosed=True, color=(0, 255, 0), thickness=2)
    
#     # For each active track, mark a small dot and label on the canvas.
#     for t in active_tracks:
#         # Get the track's bounding box and compute bottom-center.
#         box = t.to_tlbr()  # [x1, y1, x2, y2]
#         bottom_x = (box[0] + box[2]) / 2
#         bottom_y = box[3]
#         # Check if the detection is inside the video-drawn court
#         if cv2.pointPolygonTest(video_polygon, (bottom_x, bottom_y), False) >= 0:
#             pt_video = np.array([[bottom_x, bottom_y]], dtype=np.float32)
#             pt_canvas = transform_points(pt_video, H)
#             # Draw a small black dot at the transformed point
#             cv2.circle(canvas_current, (int(pt_canvas[0][0]), int(pt_canvas[0][1])), DOT_RADIUS, DOT_COLOR, DOT_THICKNESS)
#             # Draw the unique ID next to the dot
#             cv2.putText(canvas_current, f"ID:{t.track_id}", (int(pt_canvas[0][0])+5, int(pt_canvas[0][1])-5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    
#     # Overlay the canvas (with court and dot markers) onto the top-left corner of the frame.
#     output_frame = frame_copy.copy()
#     canvas_h, canvas_w = canvas_current.shape[:2]
#     if output_frame.shape[0] >= 20 + canvas_h and output_frame.shape[1] >= 20 + canvas_w:
#         output_frame[20:20+canvas_h, 20:20+canvas_w] = canvas_current
    
#     # Optionally, annotate the frame with detection boxes (for visualization)
#     annotated_frame = output_frame.copy()
#     annotated_frame = sv.BoxAnnotator(thickness=2, color=sv.Color.from_hex('#00FF00')).annotate(scene=annotated_frame, detections=detections)
    
#     cv2.imshow("Output", annotated_frame)
#     out_writer.write(annotated_frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# out_writer.release()
# cv2.destroyAllWindows()



import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO
import supervision as sv
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics.nn.tasks import DetectionModel  # Custom DetectionModel class

# --- Setup Environment & Model ---
torch.serialization.add_safe_globals([DetectionModel])
PLAYER_DETECTION_MODEL = YOLO(r"Models/volleyPlayers.pt")
# Set YOLO thresholds
PLAYER_DETECTION_MODEL.conf = 0.3  
PLAYER_DETECTION_MODEL.iou  = 0.5  

# Define the desired class for players (assuming player class is 0)
DESIRED_CLASS = 0

# --- Input Paths ---
VIDEO_PATH = r"Videos/Video8.mp4"       # Video used for detection and court drawing (first frame)
BACKGROUND_IMG_PATH = "input1.png"        # Background image (canvas)
OUTPUT_VIDEO_PATH = "final_output.mp4"    # Final output video

# === STEP 1: Draw Court on Video's First Frame ===
cap = cv2.VideoCapture(VIDEO_PATH)
ret, first_frame = cap.read()
if not ret:
    raise Exception("Could not read first frame from video.")
first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
cap.release()

plt.figure(figsize=(10,6))
plt.imshow(first_frame_rgb)
plt.title("VIDEO: Draw Court - Click 4 Points")
video_polygon = []
for _ in range(4):
    pt = plt.ginput(1, timeout=0)[0]
    video_polygon.append(pt)
    plt.scatter(pt[0], pt[1], color='red', marker='o')
    if len(video_polygon) > 1:
        plt.plot([video_polygon[-2][0], video_polygon[-1][0]],
                 [video_polygon[-2][1], video_polygon[-1][1]], 'r-', lw=2)
# Close the polygon
plt.plot([video_polygon[-1][0], video_polygon[0][0]],
         [video_polygon[-1][1], video_polygon[0][1]], 'r-', lw=2)
video_polygon = np.array(video_polygon, dtype=np.float32)
plt.title("Close window when done.")
plt.show()

plt.figure(figsize=(10,6))
plt.imshow(first_frame_rgb)
plt.title("VIDEO: Draw Mid-Line - Click 2 Points")
video_halfline = np.array(plt.ginput(2, timeout=0), dtype=np.float32)
plt.scatter(video_halfline[:,0], video_halfline[:,1], color='blue', marker='x')
plt.plot(video_halfline[:,0], video_halfline[:,1], 'b-', lw=2)
plt.title("Close window when done.")
plt.show()

# === STEP 2: Use Entire Background Image as Canvas ===
background_img = cv2.imread(BACKGROUND_IMG_PATH)
if background_img is None:
    raise Exception("Could not load background image.")
# Use the entire background image as the canvas.
bg_h, bg_w = background_img.shape[:2]
background_polygon = np.array([[0, 0], [bg_w, 0], [bg_w, bg_h], [0, bg_h]], dtype=np.float32)

# === STEP 3: Compute Homography from Video Court to Background Canvas ===
# This maps points from the video-drawn court (video_polygon) to the entire background image.
H, _ = cv2.findHomography(video_polygon, background_polygon)

# === STEP 4: Prepare for Tracking & Dot Drawing ===
# Dot parameters for marking players on the canvas
DOT_RADIUS = 3
DOT_COLOR = (0, 0, 0)  # Black
DOT_THICKNESS = -1     # Filled circle

# Initialize DeepSORT tracker (to assign unique IDs)
tracker = DeepSort(
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

def transform_points(points, H):
    pts = cv2.perspectiveTransform(points.reshape(-1, 1, 2), H)
    return pts.reshape(-1, 2).astype(int)

# === STEP 5: Process Video Frames, Detect, Track, and Overlay Canvas ===
cap = cv2.VideoCapture(VIDEO_PATH)
fps_video = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps_video, (width, height))

cv2.namedWindow("Output", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_copy = frame.copy()
    
    # Run YOLO detection on the frame
    result = PLAYER_DETECTION_MODEL.predict(frame, conf=0.1)[0]
    detections = sv.Detections.from_ultralytics(result)
    
    # Filter detections: only keep those of the desired player class.
    filtered_boxes = []
    filtered_confidences = []
    filtered_class_ids = []
    for i, box in enumerate(detections.xyxy):
        if int(detections.class_id[i]) != DESIRED_CLASS:
            continue
        filtered_boxes.append(box)
        filtered_confidences.append(detections.confidence[i])
        filtered_class_ids.append(detections.class_id[i])
    if len(filtered_boxes) > 0:
        detections.xyxy = np.array(filtered_boxes)
        detections.confidence = np.array(filtered_confidences)
        detections.class_id = np.array(filtered_class_ids)
    else:
        detections = sv.Detections.empty()
    
    # Update tracker with filtered detections to assign unique IDs.
    if len(detections) > 0:
        results_list = []
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
    
    # Create a copy of the background canvas (the entire background image) for this frame.
    canvas_current = background_img.copy()
    # (Optionally) Draw the boundary of the background canvas
    bg_poly = background_polygon.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(canvas_current, [bg_poly], isClosed=True, color=(0, 255, 0), thickness=2)
    
    # For each active track, mark a small black dot on the canvas.
    for t in active_tracks:
        box = t.to_tlbr()  # [x1, y1, x2, y2]
        bottom_x = (box[0] + box[2]) / 2
        bottom_y = box[3]
        # Use video-drawn court for filtering
        if cv2.pointPolygonTest(video_polygon, (bottom_x, bottom_y), False) >= 0:
            pt_video = np.array([[bottom_x, bottom_y]], dtype=np.float32)
            pt_canvas = transform_points(pt_video, H)
            cv2.circle(canvas_current, (int(pt_canvas[0][0]), int(pt_canvas[0][1])), DOT_RADIUS, DOT_COLOR, DOT_THICKNESS)
    
    # Overlay the canvas (with court and dots) onto the top-left corner of the frame.
    output_frame = frame_copy.copy()
    canvas_h, canvas_w = canvas_current.shape[:2]
    if output_frame.shape[0] >= 20 + canvas_h and output_frame.shape[1] >= 20 + canvas_w:
        output_frame[20:20+canvas_h, 20:20+canvas_w] = canvas_current
    
    # Now, annotate the frame with detection boxes and unique IDs (for the players)
    if active_tracks:
        tracked_detections = sv.Detections(
            xyxy=np.array([t.to_tlbr() for t in active_tracks]),
            confidence=np.array([t.det_conf for t in active_tracks]),
            class_id=np.array([t.det_class for t in active_tracks]),
            tracker_id=np.array([t.track_id for t in active_tracks])
        )
    else:
        tracked_detections = sv.Detections.empty()
    
    annotated_frame = output_frame.copy()
    annotated_frame = sv.BoxAnnotator(thickness=2, color=sv.Color.from_hex('#00FF00')).annotate(scene=annotated_frame, detections=tracked_detections)
    labels = [f"ID: {tid}" for tid in tracked_detections.tracker_id] if len(tracked_detections) > 0 else []
    annotated_frame = sv.LabelAnnotator(text_color=sv.Color.from_hex('#FF0000'),
                                          color=sv.Color.from_hex('#00FF00')).annotate(scene=annotated_frame, detections=tracked_detections, labels=labels)
    
    cv2.imshow("Output", annotated_frame)
    out_writer.write(annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out_writer.release()
cv2.destroyAllWindows()
