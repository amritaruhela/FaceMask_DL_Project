# 😷 Face Mask Detection using Deep Learning

This project focuses on building a real-time face mask detection system using Deep Learning and Computer Vision. The idea is simple but impactful — detect whether a person is wearing a mask or not using a webcam feed.

I developed this project as part of my learning in Deep Learning, where I trained a Convolutional Neural Network (CNN) model from scratch and then deployed it for real-time prediction.

---

## 🚀 What this project does

* Detects human faces using OpenCV
* Classifies each face as **Mask 😷** or **No Mask ❌**
* Works in real-time using webcam
* Provides visual output with bounding boxes and labels

---

## 🧠 How it works

The system works in two main stages:

### 1. Face Detection

First, faces are detected using Haar Cascade Classifier provided by OpenCV.

### 2. Mask Classification

Each detected face is passed to a trained CNN model, which predicts:

* Mask
* No Mask

---

## 🛠️ Technologies Used

* Python
* OpenCV (for face detection & webcam handling)
* TensorFlow / Keras (for building and training CNN)
* NumPy
* Matplotlib

---

## 📂 Project Structure

```
FaceMask_DL_Project/
│
├── train.py                # Code for training the CNN model
├── detect.py               # Real-time mask detection using webcam
├── mask_model.h5           # Trained deep learning model
├── haarcascade_frontalface_default.xml   # Face detection model
├── dataset/
│   ├── with_mask/
│   └── without_mask/
```

---

## ▶️ How to run the project

### Step 1: Install dependencies

```
pip install tensorflow opencv-python numpy matplotlib scikit-learn
```

### Step 2: Train the model (optional if model already exists)

```
python train.py
```

### Step 3: Run real-time detection

```
python detect.py
```

---

## 📊 Results

* Training Accuracy: ~99%
* Validation Accuracy: ~94%

The model performs well on unseen data, although slight overfitting is observed.

---

## 📸 Output

The system shows:

* Green box → Mask detected
* Red box → No Mask detected

(You can add screenshots here for better visualization)

---

## 💡 What I learned

* How to build and train a CNN model
* Working with image datasets
* Real-time computer vision using OpenCV
* Integrating Deep Learning models into applications

---

## 🚀 Future Improvements

* Reduce overfitting using Dropout or Data Augmentation
* Use pre-trained models like MobileNet for better performance
* Deploy the model on edge devices or web apps

---

## 📌 Note

The dataset used includes images of people with and without masks,
organized into respective folders for training and validation.

---

## 🙌 Final Thoughts

This project helped me understand the complete pipeline of a Deep Learning application — from training to real-time deployment. It’s a great example of how AI can be applied to solve real-world problems.

---
