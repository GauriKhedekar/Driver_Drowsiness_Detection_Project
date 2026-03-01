import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import sys

# Initialize Mediapipe components
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    refine_landmarks=True
)

# Thresholds
EAR_THRESHOLD = 0.2
MAR_THRESHOLD = 0.4 
CONSEC_FRAMES = 5

# Counters
frame_count = 0
yawn_count = 0
eye_closed_count = 0
yawn_times = 0
rectangle_count = 0  # Track rectangle count

# Indices for landmarks
left_eye_indices = [33, 160, 158, 133, 153, 144]
right_eye_indices = [362, 385, 387, 263, 373, 380]
# corners and mid-points of lips
mouth_indices = [61, 291, 13, 14]

# Calculate EAR (Eye Aspect Ratio)
def calculate_EAR(landmarks, indices, frame_width, frame_height):
    eye = np.array([(landmarks[i].x * frame_width, landmarks[i].y * frame_height) for i in indices])
    hor_distance = np.linalg.norm(eye[0] - eye[3])
    ver_distance1 = np.linalg.norm(eye[1] - eye[5])
    ver_distance2 = np.linalg.norm(eye[2] - eye[4])
    return (ver_distance1 + ver_distance2) / (2.0 * hor_distance)

# Calculate MAR (Mouth Aspect Ratio)
def calculate_MAR(landmarks, indices, frame_width, frame_height):
    # Get mouth landmarks
    mouth = np.array([(landmarks[i].x * frame_width, landmarks[i].y * frame_height) for i in indices])
    # Horizontal distance (corners of the mouth)
    hor_distance = np.linalg.norm(mouth[0] - mouth[1])
    # Vertical distance (upper lip to lower lip)
    ver_distance = np.linalg.norm(mouth[2] - mouth[3])
    return ver_distance / hor_distance


# Create a Tkinter window
window = tk.Tk()
window.title("Driver Drowsiness and Yawning Detection")  # Title at the top

# Set the background color of the window
window.configure(bg="lightblue")

# Create a frame for status and counts
status_frame = tk.Frame(window, bg="lightblue")
status_frame.pack(pady=10)

# Title Label
title_label = Label(status_frame, text="Driver Drowsiness and Yawning Detection", font=("Helvetica", 16), bg="lightblue")
title_label.grid(row=0, column=0, pady=10)

# Status label
status_label = Label(status_frame, text="Status: Waiting for detection...", font=("Helvetica", 14), bg="lightblue")
status_label.grid(row=1, column=0)

# Eye closed count label
eye_closed_label = Label(status_frame, text="Eye Closed Count: 0", font=("Helvetica", 12), bg="lightblue")
eye_closed_label.grid(row=2, column=0)

# Yawn count label
yawn_label = Label(status_frame, text="Yawning Count: 0", font=("Helvetica", 12), bg="lightblue")
yawn_label.grid(row=3, column=0)

# Create a frame for the Start/Stop Button
button_frame = tk.Frame(window, bg="lightblue")
button_frame.pack(pady=10)

# Start/Stop Button
start_button_text = tk.StringVar()
start_button_text.set("Start Live")

def toggle_live_detection():
    global is_live
    if is_live:
        is_live = False
        start_button_text.set("Start Live")
    else:
        is_live = True
        start_button_text.set("Stop Live")
        
start_button = tk.Button(button_frame, textvariable=start_button_text, command=toggle_live_detection, font=("Helvetica", 14), bg="lightgreen")
start_button.pack()

# Exit Button
def exit_application():
    cap.release()
    cv2.destroyAllWindows()
    window.quit()

exit_button = tk.Button(button_frame, text="Exit", command=exit_application, font=("Helvetica", 14), bg="red")
exit_button.pack(pady=5)

# Initialize camera
cap = cv2.VideoCapture(0)

# Flag for live detection
is_live = False

# Function to update the live image in Tkinter
def update_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=im_pil)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

# Create a label to display the video stream
video_label = Label(window)
video_label.pack(pady=10)

# Main loop
while True:
    if is_live:
        success, frame = cap.read()
        if not success or frame is None:
            print("Warning: Empty or invalid frame. Skipping...")
            sys.exit(1)

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            frame_height, frame_width, _ = frame.shape

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # EAR Calculation
                    left_EAR = calculate_EAR(face_landmarks.landmark, left_eye_indices, frame_width, frame_height)
                    right_EAR = calculate_EAR(face_landmarks.landmark, right_eye_indices, frame_width, frame_height)
                    EAR = (left_EAR + right_EAR) / 2.0

                    # MAR Calculation
                    MAR = calculate_MAR(face_landmarks.landmark, mouth_indices, frame_width, frame_height)


                    # Set font properties for text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    font_thickness = 2
                    color_active = (0, 255, 0)  # Green for active
                    color_drowsy = (0, 0, 255)  # Red for drowsy

                    # Text y-offset for multiple lines
                    text_y = 40
                    status = None  # Variable to track the current status

                    # Check drowsiness (eye closed)
                    if EAR < EAR_THRESHOLD:
                        frame_count += 1
                        if frame_count >= CONSEC_FRAMES:
                            status = "DROWSY - Eyes Closed!"
                            eye_closed_count += 1
                            eye_closed_label.config(text=f"Eye Closed Count: {eye_closed_count}")
                            cv2.putText(frame, status, (10, text_y), font, font_scale, color_drowsy, font_thickness)
                            text_y += 30
                    else:
                        frame_count = 0

                    # Check yawning
                    if MAR > MAR_THRESHOLD:
                        yawn_count += 1
                        if yawn_count >= 3:
                            status = "DROWSY - Yawning!"
                            yawn_times += 1
                            yawn_label.config(text=f"Yawning Count: {yawn_times}")
                            cv2.putText(frame, status, (10, text_y), font, font_scale, color_drowsy, font_thickness)
                            text_y += 30
                    else:
                        yawn_count = 0

                    # If no drowsiness is detected, display "Active"
                    if status is None:
                        status = "Active"
                        cv2.putText(frame,f"Status: {status}", (10, text_y), font, font_scale, color_active, font_thickness)

                    # Update the GUI label
                    status_label.config(text=f"Status: {status}")

                    # Draw landmarks
                    for idx in left_eye_indices + right_eye_indices + mouth_indices:
                        x = int(face_landmarks.landmark[idx].x * frame_width)
                        y = int(face_landmarks.landmark[idx].y * frame_height)
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                    # Draw rectangles
                    for face_landmarks in results.multi_face_landmarks:
                        x_min = int(min([p.x for p in face_landmarks.landmark]) * frame_width)
                        x_max = int(max([p.x for p in face_landmarks.landmark]) * frame_width)
                        y_min = int(min([p.y for p in face_landmarks.landmark]) * frame_height)
                        y_max = int(max([p.y for p in face_landmarks.landmark]) * frame_height)
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0,255), 2)
                        rectangle_count += 1
                        title_label.config(text=f"Driver Drowsiness and Yawning Detection")

            else:
                status_label.config(text="Status: No Face Detected")

            # Update the Tkinter window with the new frame
            update_frame(frame)

        except Exception as e:
            break
    try:
        window.update()
        # Exit condition for the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit_application()
    except:
        pass
    
cap.release()
cv2.destroyAllWindows()
window.quit()
