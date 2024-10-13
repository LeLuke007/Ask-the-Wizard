import cv2, os , random, threading, csv
import numpy as np
import speech_recognition as sr
import mediapipe as mp

face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
cmdFile = 'cmd.csv'

with open(cmdFile, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['commands'])

cap = cv2.VideoCapture(0)
recognizer = sr.Recognizer()
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

listening_thread = None
current_background = None

mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

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
    return bg_resized

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
    # cv2.imshow('Frame',frame)
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=6)
    results = selfie_segmentation.process(frame)
    mask = results.segmentation_mask
    
    if len(faces) > 0:
        start_listening()

    cmd = latest_cmd().strip()
    if cmd == "change background":
        current_background = change_background(frame)
        cmd_writer([''])
    
    if current_background is not None:
        condition = mask > 0.5
        frame = np.where(condition[:, :, None], frame, current_background)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
    
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()