import cv2, os , random, threading, csv
# import numpy as np
import speech_recognition as sr

face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
cmdFile = 'cmd.csv'

if not os.path.exists(cmdFile):
    with open(cmdFile, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['command'])

cap = cv2.VideoCapture(0)
recognizer = sr.Recognizer()
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

listening_thread = None

def cmd_writer(data, cmdFile=cmdFile):
    with open(cmdFile, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)

def latest_cmd(cmdFile=cmdFile):  
    with open(cmdFile, 'r') as f:
        last_line = f.readlines()[-1]
    return last_line

def apply_filter(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def change_background(frame):
    img = random.choice(os.listdir('./bg'))
    bg_img = cv2.imread(os.path.join('./bg',img))
    bg_resized = cv2.resize(bg_img, dsize=(frame.shape[1], frame.shape[0]))
    return cv2.addWeighted(frame, 0.5, bg_resized, 0.5, 0)

def listen_for_commands():
    with sr.Microphone() as source:
        print("Listening for commands...")
        try:
            audio = recognizer.listen(source, timeout=5)
            command = recognizer.recognize_google(audio).lower()
            print(f"You said: {command}")
            if command in ["change reality", "change background"]:
                cmd_writer(['change background'])
                command = ''
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError:
            print("Could not request results")
        except sr.WaitTimeoutError:
            print("Listening timed out")

def start_listening():
    global listening_thread
    if listening_thread is None or not listening_thread.is_alive():
        listening_thread = threading.Thread(target=listen_for_commands, daemon=True)
        listening_thread.start()

while True:
    _, frame = cap.read()
    cv2.imshow('Frame',frame)
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=6)
    
    if len(faces) > 0:
        start_listening()

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        
        cmd = latest_cmd().strip()
        if cmd == "change background":
            frame = change_background(frame)

        cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()