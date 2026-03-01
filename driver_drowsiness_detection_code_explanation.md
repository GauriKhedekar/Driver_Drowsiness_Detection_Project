### Explanation of driver_drowsiness_detection.py code:-


```python
# ================================
# DRIVER DROWSINESS DETECTION SYSTEM
# Using OpenCV + MediaPipe Face Mesh
# ================================

# -------------------------------
# IMPORTING LIBRARIES
# -------------------------------

import cv2  
# cv2 is OpenCV library
# Used for computer vision tasks like:
# - Accessing webcam
# - Reading frames
# - Drawing on frames
# - Showing video window

import mediapipe as mp  
# mediapipe is Google’s ML library
# We rename it as "mp"
# 'as' keyword gives an alias (short name)

import time  
# time module is used to track how long eyes stay closed


# -------------------------------
# ACCESSING MEDIAPIPE SOLUTIONS
# -------------------------------

mp_face_mesh = mp.solutions.face_mesh
# mp → mediapipe
# solutions → prebuilt ML solutions
# face_mesh → facial landmark detector (468 face points)

mp_drawing = mp.solutions.drawing_utils
# drawing_utils helps to draw face landmarks


# -------------------------------
# INITIALIZE FACE MESH
# -------------------------------

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,          # Detect only 1 face
    refine_landmarks=True,    # Better eye landmarks
    min_detection_confidence=0.5,  # Minimum confidence to detect face
    min_tracking_confidence=0.5    # Confidence to track face
)

# FaceMesh() is a constructor
# It creates an object that detects face landmarks


# -------------------------------
# START WEBCAM
# -------------------------------

cap = cv2.VideoCapture(0)
# VideoCapture(0) → 0 means default webcam
# If you had external camera → use 1


# -------------------------------
# VARIABLES FOR DROWSINESS LOGIC
# -------------------------------

closed_eyes_start_time = None
# Will store time when eyes close

drowsy_threshold = 2
# If eyes closed for more than 2 seconds → drowsy


# -------------------------------
# MAIN LOOP (RUNS CONTINUOUSLY)
# -------------------------------

while True:
    ret, frame = cap.read()
    # cap.read() returns:
    # ret → True/False (camera working or not)
    # frame → image captured

    if not ret:
        break
        # If camera fails, stop loop


    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # OpenCV reads image in BGR format
    # MediaPipe requires RGB format


    # Process the frame using face mesh
    result = face_mesh.process(rgb_frame)
    # process() runs ML model on image


    # Check if face detected
    if result.multi_face_landmarks:
        # multi_face_landmarks → list of detected faces

        for face_landmarks in result.multi_face_landmarks:

            # Draw face landmarks on screen
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION
            )
            # frame → image
            # face_landmarks → detected points
            # FACEMESH_TESSELATION → full face mesh lines


            # -------------------------------
            # EYE LANDMARK INDEXES
            # -------------------------------

            left_eye_top = face_landmarks.landmark[159]
            left_eye_bottom = face_landmarks.landmark[145]

            # landmark[index]
            # landmark is list of 468 points
            # Each landmark has:
            # x → horizontal position
            # y → vertical position
            # z → depth

            # Calculate eye distance
            eye_distance = abs(left_eye_top.y - left_eye_bottom.y)
            # abs() → absolute value
            # If distance small → eye closed


            # -------------------------------
            # DROWSINESS LOGIC
            # -------------------------------

            if eye_distance < 0.01:
                # If eyes nearly closed

                if closed_eyes_start_time is None:
                    closed_eyes_start_time = time.time()
                    # time.time() → current timestamp

                else:
                    elapsed_time = time.time() - closed_eyes_start_time
                    # Calculate how long eyes closed

                    if elapsed_time > drowsy_threshold:
                        cv2.putText(
                            frame,
                            "DROWSY ALERT!",
                            (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            3
                        )
                        # putText() syntax:
                        # frame → image
                        # text → string to display
                        # (50,100) → position
                        # FONT_HERSHEY_SIMPLEX → font style
                        # 1 → font size
                        # (0,0,255) → color (Red in BGR)
                        # 3 → thickness

            else:
                closed_eyes_start_time = None
                # Reset if eyes open


    # Show window
    cv2.imshow("Driver Drowsiness Detection", frame)
    # imshow(window_name, image)


    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        # waitKey(1) → waits 1 millisecond
        # ord('q') → ASCII value of q
        # If q pressed → stop


# -------------------------------
# RELEASE CAMERA & CLOSE WINDOWS
# -------------------------------

cap.release()
# Releases webcam

cv2.destroyAllWindows()
# Closes all OpenCV windows
```
