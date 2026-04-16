import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("mask_model.h5")

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))
        face = face / 255.0
        face = np.reshape(face, (1,100,100,3))

        pred = model.predict(face)

        if pred[0][0] > pred[0][1]:
            label = "Mask"
            color = (0,255,0)
        else:
            label = "No Mask"
            color = (0,0,255)

        cv2.putText(frame, label, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)

    cv2.imshow("Mask Detection (DL Model)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()