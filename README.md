# Smart Student Watch – AI-Powered Student Monitoring System

Smart Student Watch is a real-time AI-powered system that monitors students' emotion, posture, attention, and drowsiness using computer vision and deep learning. The system provides real-time alerts, logs student activity, and generates behavior reports.

## 🔍 Features

- Real-time emotion detection (Happy, Neutral, Sad, etc.)
- Posture analysis (Straight vs Slouching)
- Eye closure and yawning detection for drowsiness
- Inattention tracking with alerts
- Text-to-speech voice feedback for warnings
- Face recognition for student identification
- Per-minute attendance & behavior logging (CSV)
- Report analysis via `recognition_report.py`

##  Technologies Used

| Component        | Technology                          |
|------------------|--------------------------------------|
| Backend          | Python, Flask                        |
| Computer Vision  | OpenCV, MediaPipe                    |
| Emotion Model    | TensorFlow (MobileNet)               |
| Face Recognition | face_recognition (dlib)              |
| Voice Feedback   | pyttsx3                              |
| Reports          | CSV, Matplotlib (Backend only)       |
| Frontend         | Custom HTML/CSS/JS (Not Vite)        |

---

##  Folder Structure

smart-student-watch/
│
├── app.py                    # Main Flask backend (runs the app)
├── camera.py                 # Real-time detection logic (face/emotion/posture)
├── recognition_report.py     # Summarizes CSV logs into report
├── best_mobilenet_model.h5   # Trained emotion detection model
├── face_model.h5             # (Optional) Face recognition model
├── requirements.txt          # Python dependency list
├── README.md                 # Project documentation
├── index.html                # Main frontend entry point (custom HTML)
│
├── face_data/                # Registered student face encodings
│   └── (student_image_data.jpg etc.)
│
├── reports/                  # Auto-generated student logs (.csv per student)
│   └── student_id_1.csv
│   └── student_id_2.csv
│
├── public/                   # Static assets (CSS/JS/images if needed)
│   └── style.css
│   └── script.js
│
├── models/                   # Model folder if separated
│   └── best_mobilenet_model.h5
│
└── venv/                     # Python virtual environment (ignored in Git)


yaml
Copy
Edit

---

##  Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/nvakumar/student_monitoringAI.git
cd smart-student-watch
2. Set Up the Python Backend
bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
If you're on an M1/M2 Mac and face issues installing dlib:

bash
Copy
Edit
brew install cmake boost openblas
3. Run the Backend
bash
Copy
Edit
python3 app.py
Flask will start at: http://localhost:5000

Generating Reports
To view a summary of student behavior logs saved in the reports/ folder:

bash
Copy
Edit
python3 recognition_report.py
 Project Contributors
Name	GitHub Profile
Ajay Kumar (Lead)	nvakumar
Jasmine	Jasmine-784
Teja Pranith (Gitam CSE)	TejaPranith
A. Guna Sekhar	Sekhar-1
V. Sai Sravanthi	sra30
