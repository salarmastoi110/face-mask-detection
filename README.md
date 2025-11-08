# 😷 Face Mask Detection using Deep Learning & OpenCV

A computer vision project that detects whether a person is wearing a mask or not using a live webcam feed, built using Python, TensorFlow/Keras, OpenCV, and a CNN-based model.

---

## 📸 Features

- Detects face masks in real-time using webcam
- Uses deep learning (CNN) for classification
- Trained on custom dataset
- OpenCV for face detection
- Easy-to-run Python script with no complex setup

---

## 📁 Project Structure

face-mask-detection/
├── dataset/ # Training dataset (with_mask, without_mask)
├── model/
│ └── mask_detector.model # Trained Keras model
├── app.py # Main face mask detection app (webcam)
├── train_model.py # Script to train and save the model
├── requirements.txt # Required Python packages
└── README.md


---

## 🧠 Model

- **Architecture**: CNN using TensorFlow/Keras
- **Trained on**: A dataset with two classes:
  - `with_mask`
  - `without_mask`
- **Face Detection**: OpenCV's Haar cascade frontal face detector

---

## ⚙️ Installation

### 1. Clone the repo

git clone https://github.com/salarmastoi110/face-mask-detection/edit/main/README.md
cd face-mask-detection

2. Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate  # Windows
3. Install dependencies
pip install -r requirements.txt
🏃‍♂️ Run the App
python app.py
Make sure your webcam is turned on. A window will open showing real-time detection.

🎓 Train the Model
python train_model.py
This will train the CNN on the dataset and save the model to model/mask_detector.model.

📝 Requirements
Python 3.7+
TensorFlow / Keras
OpenCV
NumPy
Matplotlib
Install via:
pip install tensorflow keras opencv-python imutils matplotlib numpy
📜 License
MIT License

👨‍💻 Author
Salar Mastoi
Educational project for real-time face mask detection using AI.
