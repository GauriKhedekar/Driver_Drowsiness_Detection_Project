import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import sys

# ================================
# INITIALIZE MEDIAPIPE
# ================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# ================================
# THRESHOLDS
# ================================
EAR_THRESHOLD = 0.2
MAR_THRESHOLD = 0.4
CONSEC_FRAMES = 5

# ================================
# COUNTERS
# ================================
frame_count = 0
eye_closed_count = 0
yawn_times = 0

# State flags (IMPORTANT FIX)
eyes_closed_prev = False
mouth_open_prev = False

# ================================
# LANDMARK INDICES
# ================================
left_eye_indices = [33, 160, 158, 133, 153, 144]
right_eye_indices = [362, 385, 387, 263, 373, 380]
mouth_indices = [61, 291, 13, 14]

# ================================
# EAR CALCULATION
# ================================
def calculate_EAR(landmarks, indices, w, h):
    eye = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices])
    hor = np.linalg.norm(eye[0] - eye[3])
    ver1 = np.linalg.norm(eye[1] - eye[5])
    ver2 = np.linalg.norm(eye[2] - eye[4])
    return (ver1 + ver2) / (2.0 * hor)

# ================================
# MAR CALCULATION
# ================================
def calculate_MAR(landmarks, indices, w, h):
    mouth = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices])
    hor = np.linalg.norm(mouth[0] - mouth[1])
    ver = np.linalg.norm(mouth[2] - mouth[3])
    return ver / hor

# ================================
# TKINTER UI
# ================================
window = tk.Tk()
window.title("Driver Drowsiness and Yawning Detection")
window.configure(bg="lightblue")

status_frame = tk.Frame(window, bg="lightblue")
status_frame.pack(pady=10)

title_label = Label(status_frame, text="Driver Drowsiness and Yawning Detection",
                    font=("Helvetica", 16), bg="lightblue")
title_label.grid(row=0, column=0, pady=10)

status_label = Label(status_frame, text="Status: Waiting for detection...",
                     font=("Helvetica", 14), bg="lightblue")
status_label.grid(row=1, column=0)

eye_closed_label = Label(status_frame, text="Eye Closed Count: 0",
                         font=("Helvetica", 12), bg="lightblue")
eye_closed_label.grid(row=2, column=0)

yawn_label = Label(status_frame, text="Yawning Count: 0",
                   font=("Helvetica", 12), bg="lightblue")
yawn_label.grid(row=3, column=0)

button_frame = tk.Frame(window, bg="lightblue")
button_frame.pack(pady=10)

start_button_text = tk.StringVar()
start_button_text.set("Start Live")

is_live = False

def toggle_live_detection():
    global is_live
    is_live = not is_live
    start_button_text.set("Stop Live" if is_live else "Start Live")

start_button = tk.Button(button_frame, textvariable=start_button_text,
                         command=toggle_live_detection,
                         font=("Helvetica", 14), bg="lightgreen")
start_button.pack()

def exit_application():
    cap.release()
    cv2.destroyAllWindows()
    window.quit()

exit_button = tk.Button(button_frame, text="Exit",
                        command=exit_application,
                        font=("Helvetica", 14), bg="red")
exit_button.pack(pady=5)

video_label = Label(window)
video_label.pack(pady=10)

# ================================
# CAMERA
# ================================
cap = cv2.VideoCapture(0)

def update_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=im_pil)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

# ================================
# MAIN LOOP
# ================================
while True:

    if is_live:
        success, frame = cap.read()
        if not success:
            sys.exit(1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        frame_height, frame_width, _ = frame.shape
        status = None

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                left_EAR = calculate_EAR(face_landmarks.landmark,
                                         left_eye_indices,
                                         frame_width, frame_height)

                right_EAR = calculate_EAR(face_landmarks.landmark,
                                          right_eye_indices,
                                          frame_width, frame_height)

                EAR = (left_EAR + right_EAR) / 2.0

                MAR = calculate_MAR(face_landmarks.landmark,
                                    mouth_indices,
                                    frame_width, frame_height)

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2

                # ================================
                # EYE CLOSURE LOGIC (FIXED)
                # ================================
                if EAR < EAR_THRESHOLD:
                    frame_count += 1

                    if frame_count >= CONSEC_FRAMES:
                        status = "DROWSY - Eyes Closed!"

                        if not eyes_closed_prev:
                            eye_closed_count += 1
                            eye_closed_label.config(
                                text=f"Eye Closed Count: {eye_closed_count}"
                            )
                            eyes_closed_prev = True

                else:
                    frame_count = 0
                    eyes_closed_prev = False

                # ================================
                # YAWNING LOGIC (FIXED)
                # ================================
                if MAR > MAR_THRESHOLD:

                    if not mouth_open_prev:
                        yawn_times += 1
                        yawn_label.config(
                            text=f"Yawning Count: {yawn_times}"
                        )
                        mouth_open_prev = True

                    status = "DROWSY - Yawning!"

                else:
                    mouth_open_prev = False

                # ================================
                # DISPLAY STATUS
                # ================================
                if status is None:
                    status = "Active"

                status_label.config(text=f"Status: {status}")

                cv2.putText(frame, status,
                            (10, 40),
                            font, font_scale,
                            (0, 0, 255) if "DROWSY" in status else (0, 255, 0),
                            thickness)

                # Draw landmarks
                for idx in left_eye_indices + right_eye_indices + mouth_indices:
                    x = int(face_landmarks.landmark[idx].x * frame_width)
                    y = int(face_landmarks.landmark[idx].y * frame_height)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        else:
            status_label.config(text="Status: No Face Detected")

        update_frame(frame)

    try:
        window.update()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit_application()
    except:
        pass

cap.release()
cv2.destroyAllWindows()
window.quit()
