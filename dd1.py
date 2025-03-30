import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import threading
import time
import csv
from datetime import datetime
import pyttsx3

# Constants
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.75
CONSECUTIVE_FRAMES_EYE = 20
CONSECUTIVE_FRAMES_MOUTH = 15
GRAPH_LENGTH = 75

# Landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 291, 39, 181, 0, 17, 269, 405]

# GUI Parameters
WINDOW_WIDTH = 1500
WINDOW_HEIGHT = 900
SIDEBAR_WIDTH = 450
THEME_COLOR = (18, 18, 18)
ACCENT_COLOR = (0, 255, 255)
WARNING_COLOR = (0, 76, 153)
GRAPH_GLOW = (50, 255, 255)

# Initialize models
face_model = YOLO('yolov8n.pt')
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Initialize TTS
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# State variables
alarm_active = False
eye_history = []
mouth_history = []
fatigue_score = 0
log_entries = []

def create_gradient(width, height, color1, color2):
    """Create vertical gradient background"""
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        t = y / height
        gradient[y, :] = (1 - t) * np.array(color1) + t * np.array(color2)
    return gradient

def eye_aspect_ratio(landmarks, eye_indices):
    """Calculate Eye Aspect Ratio"""
    points = [np.array([landmarks[i].x, landmarks[i].y]) for i in eye_indices]
    return (np.linalg.norm(points[1]-points[5]) + np.linalg.norm(points[2]-points[4])) / (2 * np.linalg.norm(points[0]-points[3]))

def mouth_aspect_ratio(landmarks, mouth_indices):
    """Calculate Mouth Aspect Ratio"""
    points = [np.array([landmarks[i].x, landmarks[i].y]) for i in mouth_indices]
    return (np.linalg.norm(points[1]-points[7]) + np.linalg.norm(points[3]-points[5])) / (2 * np.linalg.norm(points[0]-points[4]))

def create_sidebar():
    """Create modern glassmorphic sidebar"""
    sidebar = create_gradient(SIDEBAR_WIDTH, WINDOW_HEIGHT, (30, 30, 30), (10, 10, 10))
    sidebar = cv2.GaussianBlur(sidebar, (15, 15), 0)

    
    # Glass effect
    overlay = sidebar.copy()
    cv2.rectangle(overlay, (0, 0), (SIDEBAR_WIDTH, WINDOW_HEIGHT), (255, 255, 255), -1)
    sidebar = cv2.addWeighted(overlay, 0.1, sidebar, 0.9, 0)
    
    # Title with neon effect
    cv2.putText(sidebar, "DATA", (40, 80), 
                cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), 5)
    cv2.putText(sidebar, "DATA", (40, 80), 
                cv2.FONT_HERSHEY_COMPLEX, 1.5, ACCENT_COLOR, 2)
    
    # Fatigue meter
    cv2.putText(sidebar, "FATIGUE LEVEL", (40, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.rectangle(sidebar, (40, 160), (400, 170), (50, 50, 50), 1)
    cv2.rectangle(sidebar, (40, 160), 
                 (40 + int(360 * (fatigue_score/100)), 170), 
                 (int(255 * (fatigue_score/100)), 0, 
                 255 - int(255 * (fatigue_score/100))), -1)
    
    return sidebar

def draw_spectrum_graph(data, width, height, color, threshold):
    """Create smooth gradient graph with glow effect"""
    graph = np.zeros((height, width, 3), np.uint8)
    
    if len(data) < 2:
        return graph
    
    x = np.linspace(0, width, len(data))
    y = (np.array(data) * height).clip(0, height)
    
    # Create gradient fill
    for i in range(1, len(data)):
        alpha = i/len(data)
        shade = tuple(int(c * alpha) for c in color)
        cv2.line(graph, 
                (int(x[i-1]), height - int(y[i-1])),
                (int(x[i]), height - int(y[i])),
                shade, 3)
    
    # Add glow
    glow = cv2.GaussianBlur(graph, (23, 23), 0)
    graph = cv2.addWeighted(graph, 0.7, glow, 0.3, 0)
    
    # Threshold line
    cv2.line(graph, (0, height - int(threshold * height)),
             (width, height - int(threshold * height)),
             (100, 100, 255), 2)
    
    return graph

def create_radial_gauge(value, max_value, size, color):
    """Create modern radial gauge with 3D effect"""
    gauge = np.zeros((size, size, 3), np.uint8)
    
    # Base circle
    cv2.ellipse(gauge, (size//2, size//2),
               (size//2-10, size//2-10), 0, 0, 360,
               (50, 50, 50), -1)
    
    # Value arc
    angle = 180 * (value / max_value)
    cv2.ellipse(gauge, (size//2, size//2),
               (size//2-10, size//2-10), 0, 0, angle,
               color, -1)
    
    # Glass effect
    overlay = gauge.copy()
    cv2.ellipse(overlay, (size//2, size//2),
               (size//2-5, size//2-5), 0, 0, 360,
               (255, 255, 255), -1)
    gauge = cv2.addWeighted(overlay, 0.2, gauge, 0.8, 0)
    
    # Center dot
    cv2.circle(gauge, (size//2, size//2), 5, color, -1)
    
    return gauge

def voice_alert(message):
    """Generate voice alert with queuing"""
    engine.say(message)
    engine.runAndWait()

# Main processing loop
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

frame_counter_eye = 0
frame_counter_mouth = 0
last_alert_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Create main interface
    final_frame = create_gradient(WINDOW_WIDTH, WINDOW_HEIGHT,
                                (10, 10, 20), (5, 5, 10))
    sidebar = create_sidebar()
    frame = cv2.flip(frame, 1)
    
    # Default values
    avg_ear = EAR_THRESHOLD
    mar = MAR_THRESHOLD

    # Face detection
    results = face_model(frame, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    for box, cls in zip(boxes, classes):
        if cls == 0:  # Person class
            x1, y1, x2, y2 = map(int, box)
            face_roi = frame[y1:y2, x1:x2]
            
            # Stylish face bounding box
            cv2.rectangle(final_frame, (x1+3, y1+3), (x2+3, y2+3),
                         (0, 0, 0), 2)
            cv2.rectangle(final_frame, (x1, y1), (x2, y2),
                         ACCENT_COLOR, 2)
            
            # Process face mesh
            rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            results_face = face_mesh.process(rgb_face)

            if results_face.multi_face_landmarks:
                landmarks = results_face.multi_face_landmarks[0].landmark
                
                # Calculate metrics
                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
                mar = mouth_aspect_ratio(landmarks, MOUTH)
                avg_ear = (left_ear + right_ear) / 2

                # Update history
                eye_history.append(avg_ear)
                mouth_history.append(mar)
                while len(eye_history) > GRAPH_LENGTH:
                    eye_history.pop(0)
                while len(mouth_history) > GRAPH_LENGTH:
                    mouth_history.pop(0)

                # Drowsiness detection
                if avg_ear < EAR_THRESHOLD:
                    frame_counter_eye += 1
                    fatigue_score = min(fatigue_score + 1, 100)
                    if frame_counter_eye >= CONSECUTIVE_FRAMES_EYE:
                        cv2.putText(final_frame, "DROWSINESS ALERT!",
                                  (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.7, WARNING_COLOR, 2)
                        if time.time() - last_alert_time > 5:
                            threading.Thread(target=voice_alert,
                                          args=("Critical alert! Driver fatigue detected!",)).start()
                            last_alert_time = time.time()
                else:
                    frame_counter_eye = max(0, frame_counter_eye - 1)
                    fatigue_score = max(0, fatigue_score - 0.5)

                # Yawn detection
                if mar > MAR_THRESHOLD:
                    frame_counter_mouth += 1
                    fatigue_score = min(fatigue_score + 0.8, 100)
                    if frame_counter_mouth >= CONSECUTIVE_FRAMES_MOUTH:
                        cv2.putText(final_frame, "YAWNING ALERT!",
                                  (x1, y1-60), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.7, WARNING_COLOR, 2)
                else:
                    frame_counter_mouth = max(0, frame_counter_mouth - 1)

    # Compose interface
    video_panel = cv2.resize(frame, (WINDOW_WIDTH-SIDEBAR_WIDTH, WINDOW_HEIGHT))
    video_panel = cv2.GaussianBlur(video_panel, (5, 5), 0)
    final_frame[0:WINDOW_HEIGHT, 0:WINDOW_WIDTH-SIDEBAR_WIDTH] = video_panel
    
    # Add visualizations
    eye_graph = draw_spectrum_graph(eye_history, SIDEBAR_WIDTH-40, 200,
                                  GRAPH_GLOW, EAR_THRESHOLD)
    mouth_graph = draw_spectrum_graph(mouth_history, SIDEBAR_WIDTH-40, 200,
                                     (0, 200, 100), MAR_THRESHOLD)
    
    sidebar[200:400, 20:SIDEBAR_WIDTH-20] = eye_graph
    sidebar[420:620, 20:SIDEBAR_WIDTH-20] = mouth_graph
    
    ear_gauge = create_radial_gauge(avg_ear, EAR_THRESHOLD*2, 200, ACCENT_COLOR)
    mar_gauge = create_radial_gauge(mar, MAR_THRESHOLD*1.5, 200, (0, 255, 100))
    
    sidebar[650:850, 50:250] = ear_gauge
    sidebar[650:850, 200:400] = mar_gauge
    
    final_frame[0:WINDOW_HEIGHT, WINDOW_WIDTH-SIDEBAR_WIDTH:WINDOW_WIDTH] = sidebar
    
    cv2.imshow('Advanced Drowsiness Monitor', final_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save logs and cleanup
with open('incidents.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Timestamp', 'Event Type', 'Duration'])
    writer.writerows(log_entries)
    

cap.release()
cv2.destroyAllWindows()