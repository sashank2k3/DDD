import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import threading
import pyttsx3
import speech_recognition as sr
import webbrowser
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
import time

# ========== INITIAL SETUP ========== #
# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Color Scheme
DARK_BG = "#2E3440"
MEDIUM_BG = "#3B4252"
LIGHT_BG = "#434C5E"
ACCENT = "#81A1C1"
DANGER = "#BF616A"
SUCCESS = "#A3BE8C"
TEXT = "#ECEFF4"

# EAR/MAR Configuration
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6
CONSEC_FRAMES = 20

# Global Variables
COUNTER = 0
STATUS = "Initializing..."
ear_history = deque(maxlen=100)
mar_history = deque(maxlen=100)
listening = False

# ========== FACE DETECTION SETUP ========== #
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    D = dist.euclidean(mouth[12], mouth[16])
    return (A + B + C) / (3.0 * D)

def shape_to_np(shape, dtype="int"):
    return np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=dtype)

# ========== AI ASSISTANT FUNCTIONS ========== #
def start_assistant():
    global listening
    listening = True
    threading.Thread(target=listen_for_commands, daemon=True).start()
    respond("AI Assistant activated! How can I help you?")

def stop_assistant():
    global listening
    listening = False
    respond("AI Assistant deactivated")

def listen_for_commands():
    r = sr.Recognizer()
    while listening:
        try:
            with sr.Microphone() as source:
                audio = r.listen(source, timeout=3)
                try:
                    command = r.recognize_google(audio).lower()
                    chat_text.config(state="normal")
                    chat_text.insert('end', f"\nYou: {command}\n", 'user')
                    chat_text.config(state="disabled")
                    process_command(command)
                except sr.UnknownValueError:
                    pass
        except Exception as e:
            print(f"Microphone error: {e}")

def process_command(command):
    if 'hello' in command:
        respond("Hello! How can I assist you today?")
    elif 'navigation' in command:
        respond("Opening navigation system")
        webbrowser.open("https://maps.google.com")
    elif 'emergency' in command:
        respond("Activating emergency protocols")
    elif 'stop' in command:
        stop_assistant()
    else:
        respond("Command not recognized")

def respond(text):
    chat_text.config(state="normal")
    chat_text.insert('end', f"Assistant: {text}\n", 'assistant')
    chat_text.config(state="disabled")
    chat_text.see('end')
    engine.say(text)
    engine.runAndWait()

# ========== MAIN UI SETUP ========== #
root = tk.Tk()
root.title("Next-Gen Driver Assist")
root.attributes('-fullscreen', True)

# Configure grid layout
for i in range(3):
    root.grid_rowconfigure(i, weight=1, uniform="row")
    root.grid_columnconfigure(i, weight=1, uniform="col")

# Create main frames
frames = {}
positions = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
for pos in positions:
    frame = tk.Frame(root, bg=DARK_BG, bd=2, relief="ridge")
    frame.grid(row=pos[0], column=pos[1], sticky="nsew", padx=5, pady=5)
    frames[pos] = frame

# ========== CAMERA FEEDS ========== #
# Driver Camera
cam_label = tk.Label(frames[(0,0)], bg=DARK_BG)
cam_label.pack(fill="both", expand=True)

# Road View
road_image = ImageTk.PhotoImage(Image.open("road_view.jpg").resize((640, 480)))
road_label = tk.Label(frames[(0,1)], image=road_image, bg=DARK_BG)
road_label.pack(fill="both", expand=True)

# ========== STATUS DISPLAY ========== #
status_canvas = tk.Canvas(frames[(0,2)], bg=DARK_BG, highlightthickness=0)
status_canvas.pack(fill="both", expand=True)
status_indicator = status_canvas.create_oval(50, 50, 150, 150, fill="gray", outline="")
status_text = status_canvas.create_text(100, 180, text="INITIALIZING", 
                                      fill=TEXT, font=("Arial", 14, "bold"))

# ========== VEHICLE INFO ========== #
vehicle_frame = tk.Frame(frames[(1,2)], bg=MEDIUM_BG)
vehicle_stats = [
    ("🚗 Distance", "120 km", "#8FBCBB"),
    ("⛽ Fuel", "50%", DANGER),
    ("🔋 Battery", "80%", SUCCESS)
]
for i, (label, value, color) in enumerate(vehicle_stats):
    tk.Label(vehicle_frame, text=label, bg=MEDIUM_BG, fg=TEXT, 
            font=("Arial", 12)).grid(row=i, column=0, sticky="w", padx=10)
    tk.Label(vehicle_frame, text=value, bg=MEDIUM_BG, fg=color,
            font=("Arial", 14, "bold")).grid(row=i, column=1, sticky="e", padx=10)
vehicle_frame.place(relx=0.5, rely=0.5, anchor="center")

# ========== REAL-TIME GRAPHS ========== #
# EAR Graph
ear_fig = plt.Figure(figsize=(5,2), dpi=80)
ear_ax = ear_fig.add_subplot(111)
ear_line, = ear_ax.plot([], [], color=ACCENT, linewidth=2)
ear_ax.set_title("Eye Aspect Ratio", color=TEXT, fontsize=10)
ear_ax.set_facecolor(DARK_BG)
ear_fig.patch.set_facecolor(DARK_BG)
ear_ax.tick_params(colors=TEXT)
ear_canvas = FigureCanvasTkAgg(ear_fig, master=frames[(2,0)])
ear_canvas.get_tk_widget().pack(fill="both", expand=True)

# MAR Graph
mar_fig = plt.Figure(figsize=(5,2), dpi=80)
mar_ax = mar_fig.add_subplot(111)
mar_line, = mar_ax.plot([], [], color=DANGER, linewidth=2)
mar_ax.set_title("Mouth Aspect Ratio", color=TEXT, fontsize=10)
mar_ax.set_facecolor(DARK_BG)
mar_fig.patch.set_facecolor(DARK_BG)
mar_ax.tick_params(colors=TEXT)
mar_canvas = FigureCanvasTkAgg(mar_fig, master=frames[(2,1)])
mar_canvas.get_tk_widget().pack(fill="both", expand=True)

# ========== CONTROL BUTTONS ========== #
# AI Controls (1,0)
ai_control_frame = tk.Frame(frames[(1,0)], bg=DARK_BG)
btn_start = tk.Button(ai_control_frame, text="🚀 Start AI", command=start_assistant,
                     bg=SUCCESS, fg=TEXT, font=("Arial", 12, "bold"), padx=20, pady=10)
btn_stop = tk.Button(ai_control_frame, text="🛑 Stop AI", command=stop_assistant,
                    bg=DANGER, fg=TEXT, font=("Arial", 12, "bold"), padx=20, pady=10)
btn_start.pack(fill="x", pady=5)
btn_stop.pack(fill="x", pady=5)
ai_control_frame.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.8)

# System Controls (2,2)
sys_control_frame = tk.Frame(frames[(2,2)], bg=DARK_BG)
btn_emergency = tk.Button(sys_control_frame, text="⛔ Emergency Exit", command=root.destroy,
                         bg=DANGER, fg=TEXT, font=("Arial", 12, "bold"), padx=20, pady=10)
btn_vehicle = tk.Button(sys_control_frame, text="📊 Vehicle Details", 
                       command=lambda: webbrowser.open('https://example.com'),
                       bg=ACCENT, fg=TEXT, font=("Arial", 12, "bold"), padx=20, pady=10)
btn_emergency.pack(fill="x", pady=5)
btn_vehicle.pack(fill="x", pady=5)
sys_control_frame.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.8)

# ========== AI CHAT INTERFACE ========== #
chat_frame = tk.Frame(frames[(1,1)], bg=MEDIUM_BG)
chat_text = tk.Text(chat_frame, bg=MEDIUM_BG, fg=TEXT, 
                   font=("Arial", 11), wrap="word", state="disabled")
scrollbar = ttk.Scrollbar(chat_frame, orient="vertical", command=chat_text.yview)
chat_text.configure(yscrollcommand=scrollbar.set)

chat_header = tk.Frame(chat_frame, bg=ACCENT)
tk.Label(chat_header, text="AI Copilot", bg=ACCENT, fg=TEXT,
        font=("Arial", 12, "bold")).pack(pady=5)

chat_text.tag_configure('user', foreground=TEXT, background=LIGHT_BG)
chat_text.tag_configure('assistant', foreground=TEXT, background=ACCENT)

chat_header.pack(fill="x")
chat_text.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")
chat_frame.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.95, relheight=0.9)

# ========== DRIVER MONITORING ========== #
def update_status():
    colors = {"Alert": SUCCESS, "Drowsy": DANGER, "Fatigued": "#FFA500"}
    status_canvas.itemconfig(status_indicator, fill=colors.get(STATUS, "gray"))
    status_canvas.itemconfig(status_text, text=STATUS.upper())
    root.after(1000, update_status)

def update_graphs():
    ear_line.set_data(range(len(ear_history)), list(ear_history))
    ear_ax.relim()
    ear_ax.autoscale_view()
    ear_canvas.draw()
    
    mar_line.set_data(range(len(mar_history)), list(mar_history))
    mar_ax.relim()
    mar_ax.autoscale_view()
    mar_canvas.draw()
    
    root.after(100, update_graphs)

def monitor_driver():
    global COUNTER, STATUS
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret: continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        
        for rect in rects:
            shape = predictor(gray, rect)
            shape_np = shape_to_np(shape)
            
            left_eye = shape_np[42:48]
            right_eye = shape_np[36:42]
            mouth = shape_np[48:68]
            
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2
            mar = mouth_aspect_ratio(mouth)
            
            ear_history.append(ear)
            mar_history.append(mar)
            
            if mar > MAR_THRESHOLD:
                STATUS = "Fatigued"
                COUNTER = 0
            elif ear < EAR_THRESHOLD:
                COUNTER += 1
                if COUNTER > CONSEC_FRAMES:
                    STATUS = "Drowsy"
            else:
                COUNTER = max(0, COUNTER-1)
                STATUS = "Alert" if COUNTER == 0 else STATUS
            
            # Update camera feed
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((cam_label.winfo_width(), cam_label.winfo_height()))
            img_tk = ImageTk.PhotoImage(img)
            cam_label.config(image=img_tk)
            cam_label.image = img_tk
            
        root.update_idletasks()

# ========== START APPLICATION ========== #
if __name__ == "__main__":
    # Start monitoring threads
    threading.Thread(target=monitor_driver, daemon=True).start()
    
    # Start UI updates
    update_status()
    update_graphs()
    
    # Start main loop
    root.mainloop()
