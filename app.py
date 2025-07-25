import cv2
import numpy as np
from flask import Flask, render_template, Response
from keras.models import load_model

app = Flask(__name__)

# Load the trained Keras model
model = load_model("model/mask_detector.keras")

# Define video capture object
camera = cv2.VideoCapture(0)

# Define label names based on your 2 classes
labels_dict = {0: "Mask", 1: "No Mask"}
color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            resized = cv2.resize(face, (224, 224))  # 🔥 FIXED SIZE
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 224, 224, 3))

            try:
                result = model.predict(reshaped)[0][0]
                label = 1 if result > 0.5 else 0
                color = color_dict[label]
                label_text = labels_dict[label]

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            except Exception as e:
                print(f"Prediction error: {e}")

        # Encode the frame
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)
