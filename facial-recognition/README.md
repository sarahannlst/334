# Facial Emotion Recognition Demo

A desktop Python demo that detects facial expressions (neutral, happy, angry) from your webcam and draws a colored history strip: **yellow** for happy, **red** for angry, **blue** for neutral.

Uses **OpenCV** for face detection and **dlib** 68-point facial landmarks. No cloud or API keys required.

---

## Walkthrough: Run this on your own laptop

Follow these steps in order. Use **Terminal** (Mac/Linux) or **Command Prompt** / **PowerShell** (Windows).

### 1. Get the code

**Option A – Clone with Git (if you have it):**
```bash
git clone https://github.com/sarahannlst/334.git
cd 334/facial-recognition
```

**Option B – Download ZIP:**  
Go to https://github.com/sarahannlst/334, click **Code** → **Download ZIP**, unzip, then in Terminal:
```bash
cd ~/Downloads/334-main/facial-recognition
```
(Adjust the path if you saved it somewhere else.)

---

### 2. Check Python

You need **Python 3.8 or newer**.

```bash
python3 --version
```

If that fails, install Python from https://www.python.org/downloads/ (or use your system’s package manager).

---

### 3. Create a virtual environment and install dependencies

**Mac / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `pip install` fails (e.g. for `dlib`), see the **Troubleshooting** section at the bottom.

---

### 4. Download the face landmark model (~100 MB)

The app needs a data file that isn’t in the repo.

**Option A – Browser:**  
1. Open https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2  
2. Download the file.  
3. Unzip it (double‑click or use a decompress tool). You should get `shape_predictor_68_face_landmarks.dat`.  
4. Move that file into the `facial-recognition` folder (the same folder as `main.py`).

**Option B – Terminal (Mac/Linux, one copy‑paste block):**
```bash
cd "$(dirname "$0")"
curl -L -o shape_predictor_68_face_landmarks.dat.bz2 "https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
python3 -c "import bz2; f=bz2.BZ2File('shape_predictor_68_face_landmarks.dat.bz2'); open('shape_predictor_68_face_landmarks.dat','wb').write(f.read()); f.close()"
rm shape_predictor_68_face_landmarks.dat.bz2
```

Make sure `shape_predictor_68_face_landmarks.dat` is in the same directory as `main.py`.

---

### 5. Allow camera access (Mac only)

- When you run the app, macOS may ask for camera permission.  
- If the camera never opens: **System Settings** → **Privacy & Security** → **Camera** → turn on access for **Terminal** (or **iTerm**, etc., whatever you’re using).

---

### 6. Run the app

**Mac / Linux:**
```bash
source .venv/bin/activate
python main.py
```

**Windows:**
```cmd
.venv\Scripts\activate
python main.py
```

A window should open with your webcam. If you see “NOT CALIBRATED”, press **c** and follow the calibration steps.

---

### 7. Using the app

| Key | Action |
|-----|--------|
| **c** | Calibrate: do neutral → happy → angry. For each, make the face, press **SPACE** or **ENTER** to start, hold it, then press **SPACE** or **ENTER** again. |
| **d** | Toggle debug numbers on the video. |
| **q** | Quit. |

The colored strip at the bottom only starts recording **after** you finish calibration.

---

## Requirements

- **Python** 3.8+
- **Webcam**
- **macOS:** Camera permission for Terminal (or your terminal app) in System Settings → Privacy & Security → Camera.  
- **Windows:** Camera permission when Windows prompts.

---

## Troubleshooting

- **“Could not open webcam”**  
  - Another app may be using the camera. Close it and try again.  
  - Check system camera permissions for your terminal/app.

- **“Could not load dlib shape predictor”**  
  - Ensure `shape_predictor_68_face_landmarks.dat` is in the same folder as `main.py`.

- **`pip install dlib` fails (Windows or some Linux)**  
  - Install **CMake** and a **C++ compiler** (e.g. Visual Studio Build Tools on Windows, `build-essential` on Ubuntu), then run `pip install dlib` again.  
  - Or use a pre-built wheel if available for your Python version and OS.

- **App is slow or laggy**  
  - Close other apps. The first run can be slower while the model loads.
