# 🔒 Real-Time Danger Detection System

A real-time dangerous object detection and alert system powered by YOLOv8 and OpenCV.  
The system identifies weapons (e.g., knives, guns) from webcam feed and triggers voice alerts, email notifications, and WhatsApp messages to enhance safety monitoring.  

---

## Problem Statement  
Traditional CCTV systems lack intelligent real-time alerts for dangerous objects. This project addresses public and home safety by providing AI-driven surveillance that immediately notifies users when a threat is detected.

---

## Tech Stack  
- Computer Vision: YOLOv8 (Ultralytics, PyTorch)  
- Image Processing: OpenCV  
- Alerts & Notifications:  
  - Voice Alerts → pyttsx3  
  - Email Alerts → smtplib (limited to 3 per session)  
  - WhatsApp Alerts → Twilio API (once per session)  
- Data Logging: CSV for detection history  

---

## How It Works  
1. Captures video feed from webcam/CCTV.  
2. Runs YOLOv8 inference in real-time to detect dangerous objects.  
3. If a threat is detected →  
   - ✅ *Voice alert* played  
   - ✅ *Email alert* sent with snapshot (max 3 times/session)  
   - ✅ *WhatsApp message* sent once/session  
   - ✅ Detection logged into a CSV file  

---

## 📊 Demo Flow  
```plaintext
Webcam → YOLOv8 Detection → [Danger Object Found?] → Trigger Alerts → Log Event
