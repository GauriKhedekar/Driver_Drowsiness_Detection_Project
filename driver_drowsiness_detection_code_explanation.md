### Explanation of driver_drowsiness_detection.py code:-


```python
# ================================
# IMPORTING LIBRARIES
# ================================

import cv2  
# cv2 is OpenCV library.
# It is used for computer vision tasks like:
# - Accessing webcam
# - Processing frames
# - Drawing text & shapes on screen

import numpy as np  
# numpy is used for numerical operations.
# Here we use it for:
# - Creating arrays
# - Calculating distances (norm)

import mediapipe as mp  
# mediapipe is a Google library.
# We import it as "mp" (alias).
# Alias syntax:  import library_name as short_name
# This allows us to write mp instead of mediapipe everywhere.

import tkinter as tk  
# tkinter is Python’s built-in GUI library.
# "as tk" is aliasing (short name).

from tkinter import Label  
# This imports only Label class from tkinter.
# Syntax: from module import specific_class

from PIL import Image, ImageTk  
# PIL = Python Imaging Library.
# Image converts array to image.
# ImageTk allows showing images inside tkinter GUI.

import sys  
# sys module gives system-related functions.
# We use sys.exit() to safely exit program.


# ================================
# INITIALIZE MEDIAPIPE FACE MESH
# ================================

mp_face_mesh = mp.solutions.face_mesh  
# mp.solutions → prebuilt AI solutions in MediaPipe
# face_mesh → face landmark detection module
# We store it in variable mp_face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,  # False = video mode (better tracking)
    max_num_faces=1,          # Detect only 1 face
    refine_landmarks=True     # More accurate eye & lip landmarks
)
# FaceMesh() creates the face detection object.
# Arguments inside () are called PARAMETERS.
# parameter_name=value syntax is called keyword arguments.


# ================================
# THRESHOLD VALUES
# ================================

EAR_THRESHOLD = 0.2  
# EAR = Eye Aspect Ratio threshold.
# If EAR < 0.2 → eyes considered closed.

MAR_THRESHOLD = 0.4  
# MAR = Mouth Aspect Ratio threshold.
# If MAR > 0.4 → mouth considered open (yawning).

CONSEC_FRAMES = 5  
# If eyes stay closed for 5 consecutive frames,
# then we count it as a drowsy event.


# ================================
# GLOBAL COUNTERS
# ================================

frame_count = 0  
# Counts how many frames eyes are continuously closed.

eye_closed_count = 0  
# Counts how many times eyes were closed.

yawn_times = 0  
# Counts yawns.


# STATE FLAGS (Boolean Variables)

eyes_closed_prev = False  
# Boolean variable.
# False means eyes were NOT closed previously.

mouth_open_prev = False  
# False means mouth was NOT open previously.


# ================================
# LANDMARK INDEXES
# ================================

left_eye_indices = [33, 160, 158, 133, 153, 144]  
# List of landmark indexes for left eye.
# [] means list in Python.

right_eye_indices = [362, 385, 387, 263, 373, 380]

mouth_indices = [61, 291, 13, 14]


# ================================
# FUNCTION: CALCULATE EAR
# ================================

def calculate_EAR(landmarks, indices, w, h):
# def = defines a function
# landmarks → detected face landmarks
# indices → which landmark numbers to use
# w → frame width
# h → frame height

    eye = np.array([
        (landmarks[i].x * w, landmarks[i].y * h)
        for i in indices
    ])
    # This is list comprehension syntax.
    # for i in indices → loop through each index.
    # landmarks[i].x → normalized x coordinate
    # multiply by w to convert into pixel coordinate.
    # np.array converts list into numpy array.

    hor = np.linalg.norm(eye[0] - eye[3])
    # np.linalg.norm → calculates distance.
    # eye[0] - eye[3] → subtracts two coordinate points.

    ver1 = np.linalg.norm(eye[1] - eye[5])
    ver2 = np.linalg.norm(eye[2] - eye[4])

    return (ver1 + ver2) / (2.0 * hor)
    # return sends value back to where function was called.
    # 2.0 ensures floating point division.


# ================================
# FUNCTION: CALCULATE MAR
# ================================

def calculate_MAR(landmarks, indices, w, h):

    mouth = np.array([
        (landmarks[i].x * w, landmarks[i].y * h)
        for i in indices
    ])

    hor = np.linalg.norm(mouth[0] - mouth[1])
    ver = np.linalg.norm(mouth[2] - mouth[3])

    return ver / hor


# ================================
# TKINTER GUI WINDOW
# ================================

window = tk.Tk()
# Tk() creates main window object.

window.title("Driver Drowsiness and Yawning Detection")
# .title() sets window title.

window.configure(bg="lightblue")
# .configure() changes properties.
# bg = background color.


# Create Label widget
status_label = Label(window,
                     text="Status: Waiting...",
                     font=("Helvetica", 14),
                     bg="lightblue")

status_label.pack()
# .pack() places widget inside window.


video_label = Label(window)
video_label.pack()


# ================================
# CAMERA INITIALIZATION
# ================================

cap = cv2.VideoCapture(0)
# VideoCapture(0) → 0 means default webcam.
# cap is camera object.


# ================================
# MAIN LOOP
# ================================

while True:
# Infinite loop.

    success, frame = cap.read()
    # cap.read() returns:
    # success → True/False
    # frame → image from camera

    if not success:
        sys.exit(1)
        # Exit program if camera fails.

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert BGR to RGB.
    # cv2 uses BGR by default.
    # MediaPipe expects RGB.

    results = face_mesh.process(rgb_frame)
    # Process frame and detect face landmarks.

    frame_height, frame_width, _ = frame.shape
    # frame.shape returns (height, width, channels)
    # _ means ignore third value.


    if results.multi_face_landmarks:
    # If face detected

        for face_landmarks in results.multi_face_landmarks:
        # Loop through detected faces.

            left_EAR = calculate_EAR(
                face_landmarks.landmark,
                left_eye_indices,
                frame_width,
                frame_height
            )

            right_EAR = calculate_EAR(
                face_landmarks.landmark,
                right_eye_indices,
                frame_width,
                frame_height
            )

            EAR = (left_EAR + right_EAR) / 2.0

            MAR = calculate_MAR(
                face_landmarks.landmark,
                mouth_indices,
                frame_width,
                frame_height
            )


            # ====================
            # EYE LOGIC
            # ====================

            if EAR < EAR_THRESHOLD:
            # if statement syntax:
            # if condition:

                frame_count += 1
                # += means increment by 1.

                if frame_count >= CONSEC_FRAMES:

                    if not eyes_closed_prev:
                    # not reverses boolean value.

                        eye_closed_count += 1
                        eyes_closed_prev = True

            else:
                frame_count = 0
                eyes_closed_prev = False


            # ====================
            # YAWN LOGIC
            # ====================

            if MAR > MAR_THRESHOLD:

                if not mouth_open_prev:
                    yawn_times += 1
                    mouth_open_prev = True

            else:
                mouth_open_prev = False


            # ====================
            # DISPLAY TEXT
            # ====================

            cv2.putText(
                frame,
                f"EAR: {round(EAR,2)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            # f"" is f-string (formatted string).
            # {round(EAR,2)} rounds value to 2 decimals.

    # Convert frame for tkinter
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    window.update()
    # Updates tkinter window each loop.


# Release resources
cap.release()
cv2.destroyAllWindows()
window.quit()

```
