import cv2
import time
import csv
import pyttsx3
from ultralytics import YOLO
from collections import Counter
import ssl
import smtplib
from email.message import EmailMessage
import imghdr
from twilio.rest import Client
import requests
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()

account_sid=os.getenv("TWILIO_ACCOUNT_SID")
auth_token=os.getenv("TWILIO_AUTH_TOKEN")
twilio_whatsapp_number=os.getenv("TWILIO_WHATSAPP_NUMBER")
my_whatsapp_number=os.getenv("MY_WHATSAPP_NUMBER")

EMAIL_ADDRESS=os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD=os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER=os.getenv("EMAIL_RECEIVER")

def send_whatsapp_alert(detected_labels):
    try:
        response=requests.get("https://ipinfo.io/json")
        data=response.json()
        location=f"{data.get('city','')},{data.get('region','')},{data.get('country','')}"
    except:
        location= "Location unavailable"
    message_body= f"DANGER DETECTED: {', '.join(detected_labels)}\n📍Location:{location}"
    
    client=Client(account_sid,auth_token)
    message=client.messages.create(
        from_=twilio_whatsapp_number,
        body=message_body,
        to=my_whatsapp_number
    )
    print(f"✅WhatsApp alert sent: SID{message.sid}")
    print(f"✅WhatsApp alert sent: SID{message.status}")

def send_email_alert(detected_labels, image_path):
    msg = EmailMessage()
    msg['Subject'] = "⚠️📢 Danger Detected Alert"
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = EMAIL_RECEIVER
    body = f"Danger detected: {', '.join(detected_labels)}"
    msg.set_content(body)
    with open(image_path, 'rb') as f:
        img_data = f.read()
        img_type = imghdr.what(f.name)
        img_name = f.name
    msg.add_attachment(img_data, maintype='image', subtype=img_type, filename=img_name)
    
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)

# Load YOLOv8 model
model= YOLO(r"C:\Users\weights\best.pt")
# Define danger object classes (customize as needed)
danger_classes = ['knife','gun']

# Function to speak alert
def speak_alert(text):
    engine = pyttsx3.init('sapi5')
    engine.setProperty('rate', 160)    # Optional: Adjust speaking speed
    engine.setProperty('volume', 1.0)  # Max volume
    engine.say(text)
    engine.runAndWait()

# Function to log alerts to CSV
def log_alert(detected_class):
    with open("alerts_log.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), detected_class])

# Function to save snapshot image
def save_snapshot(frame, label):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"alert_{label}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    return filename

# Start webcam
cap = cv2.VideoCapture(0)
width=int(cap.get(3))
height=int(cap.get(4))
fourcc=cv2.VideoWriter_fourcc(*'mp4v')
out=cv2.VideoWriter('DangerDetection_output.mp4',fourcc,10.0,(width,height))

print("[INFO] Starting Danger Detector... Press Q to quit.")
last_alert_time = 0
alert_cooldown = 5  # seconds between voice alerts

# New control variables for alerts
last_email_time = 0       # Track last email sent time
email_alert_count = 0     # Limit email alerts
whatsapp_sent = False     # Allow WhatsApp only once

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame=cv2.flip(frame,1)

    # Run YOLOv8 object detection
    results = model(frame, conf=0.6,iou=0.4, stream=True)

    # Collect all danger objects in current frame
    detected_dangers = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Draw box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if label in danger_classes and conf > 0.5:
                detected_dangers.append(label)

    current_time = time.time()
    if detected_dangers and current_time - last_alert_time > alert_cooldown:
        danger_counts = Counter(detected_dangers)  # Count each detected class
        alert_parts=[]
        
        snapshot_taken= False
        detected_labels=[]
        snapshot_path=None
        
        for label,count in danger_counts.items():
            item_text= f"{count} {label}" if count>1 else f"1 {label}"
            alert_parts.append(item_text)
            log_alert(label)
            if not snapshot_taken:
                snapshot_path= save_snapshot(frame,"_".join(set(detected_dangers)))
                snapshot_taken=True
            detected_labels.append(label)

        alert_message = "Warning! " + ", ".join(alert_parts) + " detected."
        speak_alert(alert_message)

        last_alert_time = current_time

        # ✅ Email alert (once every 5 minutes, max 3 times)
        if (time.time() - last_email_time > 300) and (email_alert_count < 3):
            send_email_alert(detected_labels,snapshot_path)
            last_email_time = time.time()
            email_alert_count += 1
            print(f"📧 Email alert sent ({email_alert_count}/3)")

        # ✅ WhatsApp alert
        if not whatsapp_sent:
            send_whatsapp_alert(detected_labels)
            whatsapp_sent = True
            print("📱 WhatsApp alert sent (only once per session)")

    # Show frame
    cv2.imshow("Danger Detector", frame)
    out.write(frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print("[INFO] Detector stopped")
