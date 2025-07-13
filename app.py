import os
from flask import Flask, Response, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS
import numpy as np
import face_recognition
import base64
import csv

# Local imports
from camera import gen_frames_multi, gen_frames_teacher, get_latest_status
from recognition_report import load_reports

# -------------------- Flask Setup --------------------
app = Flask(__name__, static_folder="public", static_url_path="/")
CORS(app)
app.secret_key = os.urandom(24)

# -------------------- Ensure Folders Exist --------------------
os.makedirs("face_data", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

# -------------------- Serve Frontend --------------------
@app.route("/")
def serve_index():
    return send_from_directory("public", "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory("public", path)

# -------------------- Video Feeds --------------------
@app.route("/api/video_feed")
def video_feed():
    return Response(gen_frames_multi(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/api/teacher_feed")
def teacher_feed():
    return Response(gen_frames_teacher(), mimetype='multipart/x-mixed-replace; boundary=frame')

# -------------------- Live Monitoring Status --------------------
@app.route("/api/monitoring_status")
def monitoring_status():
    return jsonify(get_latest_status())

# -------------------- Live Detected Students (for teacher) --------------------
@app.route("/api/students")
def get_students():
    status = get_latest_status()
    return jsonify({
        "students": [
            {
                "id": status["detected_student_id"],
                "name": "Unknown",  # Modify if mapping available
                "emotion": status["emotion"],
                "attention": status["attention"],
                "posture": status["posture"],
                "confidence": status["confidence"],
                "last_seen": time_now()
            }
        ] if status["detected_student_id"] != "Unknown" else []
    })

# -------------------- Register Student (base64 images) --------------------
@app.route("/api/register", methods=["POST"])
def register_student():
    data = request.get_json()
    name = data.get("name")
    reg_id = data.get("reg_id")
    images_data = data.get("imagesData", [])

    if not name or not reg_id or not images_data:
        return jsonify({"error": "Missing required fields"}), 400

    folder = os.path.join("face_data", reg_id)
    os.makedirs(folder, exist_ok=True)

    encodings = []
    for i, base64_img in enumerate(images_data):
        img_data = base64.b64decode(base64_img.split(",")[1])
        img_path = os.path.join(folder, f"{reg_id}_{i}.jpg")
        with open(img_path, "wb") as f:
            f.write(img_data)

        image = face_recognition.load_image_file(img_path)
        face_enc = face_recognition.face_encodings(image)
        if face_enc:
            encodings.append(face_enc[0])

    if encodings:
        avg_encoding = np.mean(encodings, axis=0)
        np.save(os.path.join(folder, f"{reg_id}_encoding.npy"), avg_encoding)
        return jsonify({"status": "success", "message": f"Student {reg_id} registered"})
    else:
        return jsonify({"error": "No valid face encodings found"}), 400

# -------------------- Student Login --------------------
@app.route("/api/login", methods=["POST"])
def login_student():
    data = request.get_json()
    reg_id = data.get("registration_id")

    if not reg_id:
        return jsonify({"error": "Missing registration_id"}), 400

    folder = os.path.join("face_data", reg_id)
    if not os.path.exists(folder):
        return jsonify({"error": "Student not found"}), 404

    return jsonify({
        "status": "success",
        "student_name": reg_id,  # Modify if mapping exists
        "registration_id": reg_id
    })

# -------------------- Get Reports --------------------
@app.route("/api/report")
def report():
    return jsonify({"reports": load_reports()})

@app.route("/api/download_report/<student_id>")
def download_csv(student_id):
    path = os.path.join("reports", f"{student_id}.csv")
    if os.path.exists(path):
        return send_from_directory("reports", f"{student_id}.csv", as_attachment=True)
    return jsonify({"error": "Report not found"}), 404

# -------------------- Delete All Students --------------------
@app.route("/api/delete_all_students", methods=["POST"])
def delete_all_students():
    for folder in os.listdir("face_data"):
        student_path = os.path.join("face_data", folder)
        if os.path.isdir(student_path):
            for f in os.listdir(student_path):
                os.remove(os.path.join(student_path, f))
            os.rmdir(student_path)

    for f in os.listdir("reports"):
        os.remove(os.path.join("reports", f))

    return jsonify({"status": "deleted"})

# -------------------- Helpers --------------------
from datetime import datetime
def time_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# -------------------- Run Server --------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)
