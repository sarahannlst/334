# Facial Emotion Recognition Demo

A desktop Python demo that detects facial expressions (neutral, happy, angry) from your webcam and draws a colored history strip: **yellow** for happy, **red** for angry, **blue** for neutral.

Uses **OpenCV** for face detection and **dlib** 68-point facial landmarks for expression geometry. No cloud or API keys required.

## Setup

1. **Clone and enter the repo**
   ```bash
   cd facial-recognition
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install opencv-python dlib numpy
   ```

3. **Download the dlib face landmark model** (~100MB)
   - From: https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2  
   - Decompress and place `shape_predictor_68_face_landmarks.dat` in this directory.
   - Or run once (requires SSL):
     ```bash
     python3 -c "
     import bz2, urllib.request, ssl
     url = 'https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
     ctx = ssl.create_default_context()
     with urllib.request.urlopen(url, context=ctx) as r, open('shape_predictor_68_face_landmarks.dat.bz2', 'wb') as f:
         f.write(r.read())
     with bz2.BZ2File('shape_predictor_68_face_landmarks.dat.bz2') as f_in, open('shape_predictor_68_face_landmarks.dat', 'wb') as f_out:
         f_out.write(f_in.read())
     "
     ```

## Run

```bash
source .venv/bin/activate
python main.py
```

- **c** – Calibrate (neutral → happy → angry; press SPACE/ENTER to start/stop each step).
- **d** – Toggle debug (show feature values).
- **q** – Quit.

The colored strip at the bottom only records after calibration is complete.

## Requirements

- Python 3.8+
- Webcam
- macOS: grant camera access to Terminal (or the app you run from) in System Settings → Privacy & Security → Camera.
