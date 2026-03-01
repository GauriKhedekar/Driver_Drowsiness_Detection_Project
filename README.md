# 🚗 Driver Drowsiness & Yawning Detection System

A real-time AI-based **Driver Drowsiness Detection System** built using **OpenCV** and **MediaPipe Face Mesh**.

This system detects:

- 👁 Eye Closure (Drowsiness Detection using EAR)
- 😮 Yawning (Using MAR)
- 🟥 Face Bounding Box
- 📊 Live Status Updates via Tkinter GUI

---

# 🧠 How It Works

The system uses **MediaPipe Face Mesh** to extract 468 facial landmarks and calculates:

### 🔹 EAR (Eye Aspect Ratio)
Detects if eyes remain closed for consecutive frames.

### 🔹 MAR (Mouth Aspect Ratio)
Detects yawning based on mouth opening ratio.

If:
- Eyes stay closed for a threshold duration → 🚨 Drowsy Alert  
- Yawning occurs multiple times → 🚨 Drowsy Alert  

---

# 🛠 Technologies Used

- Python 3.10.11
- OpenCV
- MediaPipe (0.10.9)
- NumPy
- Pillow
- Tkinter (GUI)

---

# 📂 Project Structure

```
Driver-Drowsiness-Detection/
│
├── driver_drowsiness_detection.py
├── requirements.txt
└── README.md
```

---

# ⚙️ COMPLETE SETUP GUIDE

Follow carefully to avoid MediaPipe errors.

---

## STEP 1️⃣ Set Python Version in VS Code


-  Open **Visual Studio Code**
  
- In the top menu bar, click:

`View`
- From the dropdown menu, click:

`Command Palette`

- In the Command Palette search box, type:

`Python: Select Interpreter`

- Click on:

`Python: Select Interpreter`

- Choose Correct Python Version

From the list, select:


`Python 3.10.11`


---

## ❗ If Python 3.10.11 is NOT available

Download and install it from:

https://www.python.org/downloads/release/python-31011/

During installation:
- ✅ Check **"Add Python to PATH"**

Then:
- Restart VS Code
- Select interpreter again

---

## STEP 2️⃣ Create Virtual Environment

Open terminal in project folder:


`python -m venv venv`

Activate it:

`venv\Scripts\activate.bat`


## STEP 3️⃣ Install Dependencies

Using requirements file:

`pip install -r requirements.txt`

OR install manually:

`pip install mediapipe==0.10.9 opencv-python numpy pillow`

## STEP 4️⃣ Run The Project


`python driver_drowsiness_detection.py`

Click Start Live to begin detection.

---


## ⚠️ Important Notes

- ❌ Do NOT use Python 3.12+

- ❌ Do NOT name any file mediapipe.py

- Always activate venv before running

- MediaPipe works best with Python 3.10.11 on Windows

---


## 🚀 Future Improvements

- 🔊 Add alarm sound when drowsy

- 📊 Log detection data

- 🎥 Record events

- 🌐 Deploy as Web App

- 📱 Convert to Mobile App

- 👩‍💻 Author
