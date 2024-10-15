import cv2, os , random, threading, csv
import numpy as np
import speech_recognition as sr
import mediapipe as mp
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
cmdFile = 'cmd.csv'

os.system('cls' if os.name == 'nt' else 'clear')
with open(cmdFile, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['commands'])

cap = cv2.VideoCapture(0)
recognizer = sr.Recognizer()
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

listening_thread = None
current_background = None
current_time_filter = None
home = True

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

def display_text(frame, text, x, y, font_size=0.5, color=(0,255,0), thickness=1):
    cv2.putText(frame, text, (x,y), cv2.FONT_HERSHEY_TRIPLEX, font_size, color, thickness, cv2.LINE_AA)

def gray_filter(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def change_background(frame):
    img = random.choice(os.listdir('./bg'))
    bg_img = cv2.imread(os.path.join('./bg',img))
    bg_resized = cv2.resize(bg_img, dsize=(frame.shape[1], frame.shape[0]))
    return bg_resized

def emoji_reaction(emotion,x,y,w,h,frame):
    if emotion == 'happy':
        emoji = cv2.imread('./img/happy.jpg')
    elif emotion == 'sad':
        emoji = cv2.imread('./img/sad.jpg')
    elif emotion == 'surprise':
        emoji = cv2.imread('./img/surprise.jpg')
    elif emotion == 'fear':
        emoji = cv2.imread('./img/fear.jpg')
    else:
        return frame
    emoji_resized = cv2.resize(emoji, dsize=(w,h))
    frame[y:y+h, x:x+w] = emoji_resized
    return frame
    
def age_filter(frame, x, y, w, h, filter_type):
    face_region = frame[y:y+h, x:x+w]
    if filter_type == "reveal":
        kernel = np.array([[0, -1, 0], 
                           [-1, 5, -1], 
                           [0, -1, 0]])
        face_region = cv2.filter2D(face_region, -1, kernel)
    
    elif filter_type == "hide":
        face_region = cv2.GaussianBlur(face_region, (51, 51), 0)
        face_region = cv2.addWeighted(face_region, 1.5, face_region, -0.5, 0)

    frame[y:y+h, x:x+w] = face_region
    return frame

def listen_for_commands():
    global current_time_filter
    with sr.Microphone() as source:
        print("Listening for commands...")
        try:
            audio = recognizer.listen(source, timeout=5)
            command = recognizer.recognize_google(audio).lower()
            print(f"You said: {command}")
            if command in ["change reality", "change background", "change dimension", 'dimension change', 'dimension', 'change my reality']:
                cmd_writer(['change background'])
                command = ''
            if command in ["change time", "time change", "change age", "age change", "age", 'change my time']:
                current_time_filter = 'reveal' if current_time_filter == 'hide' else 'hide'
                cmd_writer(['change time'])
                command = ''
            if command in ['return to home', 'reset background', 'remove background', 'home']:
                cmd_writer(['home'])
                command = ''
            if command in ['mood', 'mood check', 'what\'s my mood', "discover my mood"]:
                cmd_writer(['mood'])
                command = ''
            if command in ["exit", "quit", "run away",'runaway']:
                cmd_writer(['exit'])
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
    if home:
        fr = cv2.imread('./img/portal.png')
        (h,w,ch)=fr.shape
        display_text(fr, "ASK THE WIZARD", ((1*w)//5), 40, 1.5, (255,255,255), 2)
        display_text(fr, "Speak to the Wizard! OR Press:", 8, h-110, color=(255,255,0))
        display_text(fr, "'d' to get your dimension changed", 8, h-90)
        display_text(fr, "'t' to hide/reveal yourself", 8, h-70)
        display_text(fr, "'m' to discover your mood", 8, h-50)
        display_text(fr, "'h' to return home", 8, h-30)
        display_text(fr, "'q' to run away", 8, h-10)
    
    if len(faces) > 0:
        start_listening()

    cmd = latest_cmd().strip()
    if cmd == "change background":
        home = False
        current_background = change_background(frame)
        cmd_writer([''])

    for (x,y,w,h) in faces:
        # cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        if cmd == "mood":
            home = False
            face_emo = frame[y:y+h, x:x+w]
            r = DeepFace.analyze(face_emo, actions=['emotion'], enforce_detection=False)
            emotion = r[0]['dominant_emotion']
            frame = emoji_reaction(emotion,x,y,w,h,frame)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        
        if cmd == "change time":
            home = False
            if current_time_filter is None:
                current_time_filter = "reveal" if random.choice([True, False]) else "hide"
            frame = age_filter(frame, x, y, w, h, current_time_filter)
            cv2.putText(frame, f"Age: {current_time_filter}", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
    
    if cmd == "home":
        current_background = None
        current_time_filter = None
        home = True
        cmd_writer([''])
    
    if cmd == "exit":
        break
    
    if current_background is not None:
        condition = mask > 0.5
        frame = np.where(condition[:, :, None], frame, current_background)
    
    if home:
        cv2.imshow('Frame', fr)
    else:
        cv2.imshow('Frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('d'):
        cmd_writer(['change background'])
    elif key == ord('t'):
        current_time_filter = 'reveal' if current_time_filter == 'hide' else 'hide'
        cmd_writer(['change time'])
    elif key == ord('m'):
        cmd_writer(['mood'])
    elif key == ord('h'):
        cmd_writer(['home'])
    elif key == ord('q'):
        cmd_writer(['exit'])
        break

os.system('cls' if os.name == 'nt' else 'clear')
# credit = cv2.imread('./img/credit.png')
# cv2.imshow('Frame', credit)
# cv2.waitKey(5000)

print('Program Exited')
cap.release()
cv2.destroyAllWindows()