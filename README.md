# Drowsiness-Detection-System-u
# 💤 Drowsiness Detection System using Computer Vision

This project is a **real-time Drowsiness Detection System** developed using **Python**, **OpenCV**, and **MediaPipe**.  
It detects driver drowsiness by monitoring **eye closure** through facial landmarks and generates **visual and voice alerts**.

---

## 🚀 Features
- Real-time webcam video processing
- Face detection using MediaPipe Face Mesh
- Eye landmark detection
- Eye Aspect Ratio (EAR) based drowsiness detection
- Visual feedback:
  - Face bounding box
  - Eye bounding boxes and outlines
- Alerts:
  - On-screen text alert
  - Voice alert (system volume dependent)
- Lightweight and works on CPU (no GPU required)

---

## 🧠 How the System Works
1. Captures live video from the webcam  
2. Detects the face using MediaPipe Face Mesh  
3. Extracts eye landmarks from the detected face  
4. Calculates Eye Aspect Ratio (EAR)  
5. If eyes remain closed for a defined duration:
   - Displays **DROWSINESS ALERT** on screen
   - Triggers a voice alert

---

## 🔁 Processing Pipeline
