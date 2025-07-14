Smart Student Watch
Smart Student Watch is a real-time AI-based student monitoring system that detects and tracks multiple students using a webcam. It provides insights on emotions, posture, drowsiness, and attention span using deep learning and computer vision.

Features
Face recognition-based student identification

Emotion detection using a MobileNet model

Posture monitoring (e.g. slouching)

Eye tracking for drowsiness detection

Real-time attention tracking

Per-minute student-wise report logging (CSV)

Auto-generated HTML summary report

Flask web interface for monitoring

Project Structure
bash
Copy
Edit
smart-student-watch/
├── app.py
├── camera.py
├── recognition_report.py
├── face_data/                # Stored face encodings for students
├── models/                   # Trained model files (e.g. best_mobilenet_model.h5)
├── reports/                  # Auto-generated student reports
├── requirements.txt
├── venv/                     # Python virtual environment
Setup Instructions
Clone the repository

bash
Copy
Edit
git clone https://github.com/nvakumar/student_monitoringAI.git
cd student_monitoringAI
Create and activate a virtual environment

bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the app

bash
Copy
Edit
python app.py
View in browser
Visit http://localhost:5000

Team Members
Ajay Kumar – @nvakumar

Jasmine – @Jasmine-784

Singampalli Teja Pranith – @TejaPranith

A. Guna Sekhar – @Sekhar-1

V Sai Sravanthi – @sra30

