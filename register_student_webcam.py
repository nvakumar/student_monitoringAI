# register_student_webcam.py

import cv2
import requests

student_id = input("Enter student ID: ")
images = []

cap = cv2.VideoCapture(0)
print("📸 Press 's' to save 5 images. Press 'q' to quit early.")

while len(images) < 5:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Register - Press 's' to capture", frame)
    key = cv2.waitKey(1)

    if key == ord('s'):
        images.append(cv2.imencode('.jpg', frame)[1].tobytes())
        print(f"✅ Captured image {len(images)}/5")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if len(images) < 1:
    print("❌ No images captured.")
    exit()

# Prepare POST request
files = [('images', ('img.jpg', img, 'image/jpeg')) for img in images]
data = {'reg_id': student_id}

print("📤 Sending registration request to /api/register...")
res = requests.post("http://localhost:5050/api/register", files=files, data=data)
print("📨 Response:", res.status_code, res.json())