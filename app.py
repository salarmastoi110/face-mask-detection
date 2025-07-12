import cv2
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("model/mask_detector.model")

# Define labels
labels = ["No Mask", "Mask"]

# Start webcam
cap = cv2.VideoCapture(0)

# Loop through frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for prediction
    img = cv2.resize(frame, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)[0][0]
    label = "Mask" if prediction > 0.5 else "No Mask"
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

    # Display label on frame
    cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    cv2.rectangle(frame, (5, 5), (635, 475), color, 4)

    # Show frame
    cv2.imshow("Face Mask Detection", frame)

    # Break on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release everything
cap.release()
cv2.destroyAllWindows()
