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
├── app.py # Flask backend server
├── camera.py # Real-time detection logic
├── recognition_report.py # Report summary generator
├── best_mobilenet_model.h5 # Trained emotion classifier
├── face_model.h5 # Face recognition model
├── face_data/ # Saved student face encodings
├── reports/ # Behavior logs per student (CSV)
├── public/ # Static frontend assets
├── index.html # Frontend entry point (HTML)
├── requirements.txt # Python dependencies
├── venv/ # Virtual environment

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
