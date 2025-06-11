# Driver Monitoring System
🚗 Next-Gen Driver Monitoring System
A real-time AI-based desktop application that detects driver drowsiness and fatigue using facial landmark analysis, live video processing, and voice assistant integration.

📌 Features
👁️ Drowsiness Detection using Eye Aspect Ratio (EAR)

💤 Fatigue/Yawning Detection using Mouth Aspect Ratio (MAR)

🎥 Live Video Monitoring with OpenCV

🤖 AI Voice Assistant for commands like “emergency,” “navigation,” etc.

📊 Real-time Graphs of EAR and MAR

🧠 Built-in Driver Status Classifier: Alert, Drowsy, Fatigued

🧭 GUI built with a clean 3x3 grid layout using Tkinter

📋 Visual vehicle info dashboard: fuel, battery, distance

🌐 Voice-activated web search/navigation system

🆘 Emergency stop and exit buttons

🛠️ Technologies Used
Python 3

OpenCV – real-time video capture & processing

Dlib – facial landmark detection

Tkinter – GUI with 3x3 layout

Pyttsx3 – text-to-speech

SpeechRecognition – voice commands

Matplotlib – EAR & MAR graphs

Threading – concurrent video and AI assistant

🧠 How It Works
The app opens a fullscreen GUI with multiple panels.

cv2.VideoCapture(0) is used to capture live video of the driver.

Dlib detects 68 facial landmarks on each frame.

EAR & MAR values are calculated per frame:

EAR < 0.25 for prolonged time → Drowsy

MAR > 0.6 → Fatigued/Yawning

A voice assistant runs in parallel to respond to commands like:

"Hello" → Greeting

"Navigation" → Open Google Maps

"Emergency" → Trigger protocols

"Stop" → Turn off assistant

EAR and MAR are plotted live on graphs.

Driver status is updated and color-coded: 🟢 Alert, 🟠 Fatigued, 🔴 Drowsy.

🖼️ GUI Layout Overview (3x3 Grid)
Cam View	Road View	Status Indicator
AI Buttons	AI Chat	Vehicle Info
EAR Graph	MAR Graph	System Controls

▶️ How to Run
Install dependencies:

bash
Copy
Edit
pip install opencv-python dlib pillow numpy scipy matplotlib pyttsx3 SpeechRecognition
Download Dlib shape predictor
Place this file in the same directory:
shape_predictor_68_face_landmarks.dat
(Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

📁 DriverMonitoringSystem/
│
├── Driving_management_system.py.py
├── shape_predictor_68_face_landmarks.dat
├── road_view.jpg
├── README.md

Shashank Varanasi
