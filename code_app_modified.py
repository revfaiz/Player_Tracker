import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import supervision as sv
from tqdm import tqdm
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
from ultralytics.nn.tasks import DetectionModel  # Import the actual DetectionModel class

# Allow safe global for custom YOLO model class (only use with trusted sources)
torch.serialization.add_safe_globals([DetectionModel])

# Load YOLOv8 model using the finetuned weights
PLAYER_DETECTION_MODEL = YOLO(r"Models\Finetuned_Model\train2_best.pt")

# Video paths
SOURCE_VIDEO_PATH = r"Videos\Video8.mp4"
TARGET_VIDEO_PATH = r"volleyball_result1.mp4"

# Class IDs
PERSON_ID = 0  # Players
MAX_PLAYERS = 12  # Maximum number of players on court (6 per team)

# Add color mapping for teams
TEAM_COLORS = {
    'left': sv.ColorPalette.from_hex(['#FF0000']),  # Red for left team
    'right': sv.ColorPalette.from_hex(['#0000FF'])  # Blue for right team
}

box_annotator = sv.BoxAnnotator(
    thickness=2,
    color=sv.Color.from_hex('#00FF00')  # Single color (green) for all players
)

label_annotator = sv.LabelAnnotator(
    text_color=sv.Color.from_hex('#000000'),
    color=sv.Color.from_hex('#00FF00')  # Match box color
)

# Initialize DeepSORT tracker
tracker = DeepSort(
    max_age=30,               # Keep tracking disappeared objects longer
    n_init=3,                 # Number of detections before tracking starts
    nms_max_overlap=0.7,      # Non-maximum suppression threshold
    max_cosine_distance=0.2,  # Maximum appearance distance for matching
    nn_budget=100,            # Feature memory for appearance matching
    embedder="mobilenet",     # Feature extractor
    half=True,
    bgr=True,
    embedder_gpu=True
)

# Load video
cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Read first frame
ret, first_frame = cap.read()
cap.release()
if not ret:
    print("Error: Could not read frame.")
    exit()

first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

# ---- Matplotlib for selecting court polygon and half-line ----
plt.figure(figsize=(10, 6))
plt.imshow(first_frame)
plt.title("Select 4 Court Polygon Points (Click 4 times)")

polygon_points = []

# Interactive selection of polygon points with line drawing
for _ in range(4):
    point = plt.ginput(1, timeout=0)[0]
    polygon_points.append(point)
    plt.scatter(point[0], point[1], color='red', marker='o')
    
    # Draw lines as points are selected
    if len(polygon_points) > 1:
        plt.plot([polygon_points[-2][0], polygon_points[-1][0]], 
                 [polygon_points[-2][1], polygon_points[-1][1]], 'r-', lw=2)

# Close the polygon by connecting the last to the first point
plt.plot([polygon_points[-1][0], polygon_points[0][0]], 
         [polygon_points[-1][1], polygon_points[0][1]], 'r-', lw=2)

polygon_points = np.array(polygon_points, dtype=np.float32)

plt.title("Select 2 Half-Line Points (Click 2 times)")
halfline_points = np.array(plt.ginput(2, timeout=0), dtype=np.float32)
plt.scatter(halfline_points[:, 0], halfline_points[:, 1], color='blue', marker='x')
plt.plot(halfline_points[:, 0], halfline_points[:, 1], 'b-', lw=2)  # Draw line for half-line

plt.show()

# Define minimap size
MAP_WIDTH, MAP_HEIGHT = 200, 300  # Swapped dimensions for vertical orientation

# Define top-down minimap reference points
minimap_court = np.array([[20, 20], [180, 20], [180, 280], [20, 280]], dtype=np.float32)

# Compute homography matrix
homography_matrix, _ = cv2.findHomography(polygon_points, minimap_court)
scaled_halfline = cv2.perspectiveTransform(halfline_points.reshape(-1, 1, 2), homography_matrix).reshape(-1, 2).astype(int)

def draw_volleyball_court(map_width, map_height):
    court_img = np.zeros((map_height, map_width, 3), dtype=np.uint8)
    court_img[:] = (255, 255, 255)  # White background

    # Define court boundaries
    cv2.rectangle(court_img, (20, 20), (180, 280), (0, 255, 0), 2)

    # Draw attack lines (3m from center in each half)
    center_y = (20 + 280) // 2
    attack_line_offset = int((280 - 20) * (3 / 9))  # Scale proportionally
    cv2.line(court_img, (20, center_y - attack_line_offset), (180, center_y - attack_line_offset), (0, 0, 255), 2)
    cv2.line(court_img, (20, center_y + attack_line_offset), (180, center_y + attack_line_offset), (0, 0, 255), 2)

    return court_img

# Generate volleyball court image
volleyball_court = draw_volleyball_court(MAP_WIDTH, MAP_HEIGHT)

# Function to transform player positions onto minimap
def transform_to_minimap(points, homography_matrix):
    transformed_points = cv2.perspectiveTransform(points.reshape(-1, 1, 2), homography_matrix)
    return transformed_points.reshape(-1, 2).astype(int)

video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
video_sink = sv.VideoSink(TARGET_VIDEO_PATH, video_info=video_info)

frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
with video_sink:
    for frame in tqdm(frame_generator, total=video_info.total_frames):
        frame = cv2.resize(frame, (1280, 720))
        
        # Detect players
        result = PLAYER_DETECTION_MODEL.predict(frame, conf=0.1)[0]
        detections = sv.Detections.from_ultralytics(result)
        
        # Filter detections inside the polygon
        filtered_indices = []
        for i, box in enumerate(detections.xyxy):
            # Check multiple points along the bottom of the bounding box
            bottom_x = (box[0] + box[2]) / 2
            bottom_y = box[3]
            
            # Check multiple points along the bottom of the player
            points_to_check = [
                (bottom_x, bottom_y),  # Center bottom
                (bottom_x - 20, bottom_y),  # Left bottom
                (bottom_x + 20, bottom_y),  # Right bottom
                (bottom_x, bottom_y - 10),  # Slightly above center
                (bottom_x - 20, bottom_y - 10),  # Slightly above left
                (bottom_x + 20, bottom_y - 10)  # Slightly above right
            ]
            
            # Check if any of the points are inside the polygon
            is_inside = False
            for point in points_to_check:
                if cv2.pointPolygonTest(polygon_points, point, False) >= 0:
                    is_inside = True
                    break
            
            if is_inside:
                filtered_indices.append(i)
        
        detections = detections[np.array(filtered_indices, dtype=int)]
        
        # Convert detections to DeepSORT format
        if len(detections) > 0:
            results_list = []
            for i, box in enumerate(detections.xyxy):
                xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                w, h = xmax - xmin, ymax - ymin
                confidence = float(detections.confidence[i])
                class_id = int(detections.class_id[i])
                
                if w > 20 and h > 40:  # Minimum size check
                    results_list.append([[xmin, ymin, w, h], confidence, class_id])
            
            # Update tracker
            tracks = tracker.update_tracks(results_list, frame=frame)
            
            # Filter tracks to only include confirmed ones with recent updates
            active_tracks = [track for track in tracks if track.is_confirmed() and track.time_since_update < 5]
            
            # Convert tracks back to detections format
            if active_tracks:
                all_detections = sv.Detections(
                    xyxy=np.array([track.to_tlbr() for track in active_tracks]),
                    confidence=np.array([track.det_conf for track in active_tracks]),
                    class_id=np.array([track.det_class for track in active_tracks]),
                    tracker_id=np.array([track.track_id for track in active_tracks])
                )
            else:
                all_detections = sv.Detections.empty()
        else:
            tracks = tracker.update_tracks([], frame=frame)
            all_detections = sv.Detections.empty()
        
        # Create minimap
        minimap = volleyball_court.copy()

        # Draw court boundaries
        cv2.polylines(minimap, [minimap_court.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Draw half-line
        cv2.line(minimap, tuple(scaled_halfline[0]), tuple(scaled_halfline[1]), (255, 0, 0), 2)
        
        # Transform player positions to minimap
        if len(all_detections) > 0:
            for i, box in enumerate(all_detections.xyxy):
                bottom_x = (box[0] + box[2]) / 2
                bottom_y = box[3]
                transformed_point = transform_to_minimap(np.array([[bottom_x, bottom_y]], dtype=np.float32), homography_matrix)
                
                # Use single color (green) for all players
                player_color = (0, 255, 0)
                cv2.circle(minimap, tuple(transformed_point[0]), 5, player_color, -1)
                cv2.putText(minimap, str(all_detections.tracker_id[i]), 
                           (transformed_point[0][0] + 5, transformed_point[0][1] - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, player_color, 2)
        
        # Overlay minimap on video
        annotated_frame = frame.copy()
        
        # Only annotate if we have detections
        if len(all_detections) > 0:
            all_detections.class_id = np.zeros(len(all_detections), dtype=np.int32)
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=all_detections)
            
            # Create labels for each detection
            labels = [f"ID: {track_id}" for track_id in all_detections.tracker_id]
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=all_detections, labels=labels)
        
        # Overlay minimap on video
        annotated_frame[20:320, 20:220] = minimap
        
        video_sink.write_frame(annotated_frame)

sv.plot_image(annotated_frame)
