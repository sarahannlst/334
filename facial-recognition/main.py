import argparse
import collections
import os
import sys
import time
from datetime import datetime
from typing import Deque, Dict, Optional, Tuple

import cv2
import dlib
import numpy as np


EMOTION_COLORS: Dict[str, tuple] = {
    "none": (50, 50, 50),       # dark gray
    "neutral": (15, 106, 55),   # sage green (BGR)
    "happy": (0, 200, 255),       # marigold (BGR)
    "angry": (60, 20, 200),       # deep rose (BGR)
}


class Baseline:
    """Stores baseline values for neutral, happy, and angry expressions."""
    def __init__(self):
        # Neutral expression baselines
        self.neutral_eyebrow_height: Optional[float] = None
        self.neutral_mouth_width: Optional[float] = None
        self.neutral_mouth_height: Optional[float] = None
        self.neutral_mouth_curvature: Optional[float] = None
        self.neutral_eye_openness: Optional[float] = None
        
        # Happy expression ranges
        self.happy_mouth_width: Optional[float] = None
        self.happy_mouth_height: Optional[float] = None
        
        # Angry expression ranges
        self.angry_eyebrow_height: Optional[float] = None
        
        self.calibrated: bool = False

    def update_neutral(
        self,
        eyebrow_height: float,
        mouth_width: float,
        mouth_height: float,
        mouth_curvature: float,
        eye_openness: float,
    ):
        """Update neutral baseline values."""
        self.neutral_eyebrow_height = eyebrow_height
        self.neutral_mouth_width = mouth_width
        self.neutral_mouth_height = mouth_height
        self.neutral_mouth_curvature = mouth_curvature
        self.neutral_eye_openness = eye_openness

    def update_happy(self, mouth_width: float, mouth_height: float):
        """Update happy expression ranges."""
        self.happy_mouth_width = mouth_width
        self.happy_mouth_height = mouth_height

    def update_angry(self, eyebrow_height: float):
        """Update angry expression range."""
        self.angry_eyebrow_height = eyebrow_height

    def finalize_calibration(self):
        """Mark calibration as complete after all three expressions are calibrated."""
        self.calibrated = True


def _euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def analyze_geometric_features(
    face_roi_gray: np.ndarray,
    predictor: dlib.shape_predictor,
) -> Tuple[float, float, float, float, float]:
    """
    Analyze geometric features of the face using dlib 68-point landmarks.
    Returns: (eyebrow_height, mouth_width, mouth_height, mouth_curvature, eye_openness)
    """
    h, w = face_roi_gray.shape

    # Predict landmarks on the face ROI using a rectangle that spans the ROI
    rect = dlib.rectangle(left=0, top=0, right=w - 1, bottom=h - 1)
    shape = predictor(face_roi_gray, rect)

    # Convert landmarks to a simple list of (x, y) in ROI coordinates
    points = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

    # --- Eyebrow height (distance between eyebrows and eyes) ---
    # Left eyebrow: 17-21, left eye: 36-41
    left_brow_y = np.mean([points[i][1] for i in range(17, 22)])
    left_eye_y = np.mean([points[i][1] for i in range(36, 42)])
    left_eyebrow_dist = (left_eye_y - left_brow_y) / float(h)

    # Right eyebrow: 22-26, right eye: 42-47
    right_brow_y = np.mean([points[i][1] for i in range(22, 27)])
    right_eye_y = np.mean([points[i][1] for i in range(42, 48)])
    right_eyebrow_dist = (right_eye_y - right_brow_y) / float(h)

    # Average eyebrow distance; smaller distance = furrowed (angry)
    eyebrow_height = float((left_eyebrow_dist + right_eyebrow_dist) / 2.0)

    # --- Mouth geometry ---
    # Outer mouth: 48-59; inner: 60-67
    mouth_left = points[48]
    mouth_right = points[54]

    # Top of mouth (average of upper lip points)
    mouth_top = (
        np.mean([points[i][0] for i in (51, 52, 53)]),
        np.mean([points[i][1] for i in (51, 52, 53)]),
    )
    # Bottom of mouth (average of lower lip points)
    mouth_bottom = (
        np.mean([points[i][0] for i in (57, 58, 59)]),
        np.mean([points[i][1] for i in (57, 58, 59)]),
    )

    # Width and height normalized by face ROI size
    mouth_width = _euclidean_distance(mouth_left, mouth_right) / float(w)
    mouth_height = _euclidean_distance(mouth_top, mouth_bottom) / float(h)

    # Mouth curvature: compare center of mouth vs corners
    mouth_center = (
        np.mean([mouth_left[0], mouth_right[0]]),
        np.mean([points[62][1], points[66][1]]),  # approximate center vertically
    )
    avg_corner_y = (mouth_left[1] + mouth_right[1]) / 2.0
    # Positive curvature when center is higher (smile), normalized by face height
    mouth_curvature = float((avg_corner_y - mouth_center[1]) / float(h))

    # --- Eye openness via Eye Aspect Ratio (EAR) ---
    def eye_aspect_ratio(indices: Tuple[int, int, int, int, int, int]) -> float:
        p1, p2, p3, p4, p5, p6 = [points[i] for i in indices]
        vertical = _euclidean_distance(p2, p6) + _euclidean_distance(p3, p5)
        horizontal = _euclidean_distance(p1, p4)
        if horizontal == 0:
            return 0.0
        return float(vertical / (2.0 * horizontal))

    # Left eye: 36-41, Right eye: 42-47
    left_ear = eye_aspect_ratio((36, 37, 38, 39, 40, 41))
    right_ear = eye_aspect_ratio((42, 43, 44, 45, 46, 47))
    eye_openness = float((left_ear + right_ear) / 2.0)

    return eyebrow_height, mouth_width, mouth_height, mouth_curvature, eye_openness


def estimate_emotion_from_face_region(
    face_roi_gray: np.ndarray,
    predictor: dlib.shape_predictor,
    baseline: Optional[Baseline] = None,
    debug: bool = False,
) -> str:
    """
    Estimate emotion from facial expressions by analyzing geometric features:
    - Eyebrow position (furrowed = angry)
    - Mouth shape (smile = happy, frown = sad)
    """
    if face_roi_gray.size == 0:
        return "none"

    # Analyze geometric features based on landmarks
    eyebrow_height, mouth_width, mouth_height, mouth_curvature, eye_openness = analyze_geometric_features(
        face_roi_gray, predictor
    )

    if debug:
        print(f"Eyebrow: {eyebrow_height:.2f} | Mouth: w={mouth_width:.3f}, h={mouth_height:.3f}, curve={mouth_curvature:.3f} | Eye: {eye_openness:.3f}")
        if baseline and baseline.calibrated:
            print(f"Neutral: brow={baseline.neutral_eyebrow_height:.2f}, mouth_w={baseline.neutral_mouth_width:.3f}, h={baseline.neutral_mouth_height:.3f}")
            if baseline.happy_mouth_width:
                print(f"Happy: mouth_w={baseline.happy_mouth_width:.3f}, h={baseline.happy_mouth_height:.3f}")
            if baseline.angry_eyebrow_height:
                print(f"Angry: brow={baseline.angry_eyebrow_height:.2f}")

    # Use baseline for relative detection if calibrated
    if baseline and baseline.calibrated:
        # Calculate distances to each expression's calibrated values
        angry_eyebrow_match = False
        happy_mouth_match = False
        
        # Check eyebrow position for angry (furrowed = lower eyebrow height)
        if baseline.angry_eyebrow_height is not None:
            angry_distance = abs(eyebrow_height - baseline.angry_eyebrow_height)
            neutral_distance = abs(eyebrow_height - baseline.neutral_eyebrow_height)
            # Eyebrows closer to angry than neutral AND below neutral (furrowed)
            if angry_distance < neutral_distance and eyebrow_height < baseline.neutral_eyebrow_height:
                angry_eyebrow_match = True
        
        # Check mouth for happy (wider and taller mouth)
        if baseline.happy_mouth_width is not None and baseline.happy_mouth_height is not None:
            happy_width_distance = abs(mouth_width - baseline.happy_mouth_width)
            neutral_width_distance = abs(mouth_width - baseline.neutral_mouth_width)
            happy_height_distance = abs(mouth_height - baseline.happy_mouth_height)
            neutral_height_distance = abs(mouth_height - baseline.neutral_mouth_height)
            
            # Both width and height closer to happy than neutral
            width_closer_to_happy = happy_width_distance < neutral_width_distance
            height_closer_to_happy = happy_height_distance < neutral_height_distance
            
            if width_closer_to_happy and height_closer_to_happy:
                happy_mouth_match = True
            # Also check if mouth is significantly wider/taller than neutral (strong smile)
            elif (mouth_width > baseline.neutral_mouth_width + 0.05 and 
                  mouth_height > baseline.neutral_mouth_height + 0.02):
                happy_mouth_match = True
        
        # Decision logic with mutual exclusion
        # Happy requires: mouth is happy AND eyebrows are NOT angry
        if happy_mouth_match:
            if not angry_eyebrow_match:
                # Clear happy signal, no angry contradiction
                return "happy"
            else:
                # Both signals present - check which is stronger
                # If mouth is very close to happy range, prioritize happy
                if baseline.happy_mouth_width is not None:
                    mouth_width_diff_from_happy = abs(mouth_width - baseline.happy_mouth_width)
                    mouth_width_diff_from_neutral = abs(mouth_width - baseline.neutral_mouth_width)
                    # If mouth is much closer to happy than neutral, it's happy
                    if mouth_width_diff_from_happy < mouth_width_diff_from_neutral * 0.5:
                        return "happy"
                # Otherwise, if eyebrows are clearly angry, it might be angry
                # But prioritize happy if mouth is clearly happy
                return "happy"  # Smile is usually more obvious
        
        # Angry requires: eyebrows are angry AND mouth is NOT happy
        if angry_eyebrow_match:
            if not happy_mouth_match:
                # Clear angry signal, no happy contradiction
                return "angry"
            # If both are present, we already handled it above (prioritized happy)
        
        # Neutral: close to neutral baseline or ambiguous
        return "neutral"

    else:
        # Fallback to absolute thresholds if not calibrated
        # Based on typical values: baseline eyebrow ~0.14, angry ~0.12
        if eyebrow_height < 0.12:
            return "angry"

        # Happy: wide and tall mouth (based on debug: width >0.4, height >0.15 when smiling)
        if mouth_width > 0.35 and mouth_height > 0.14:
            return "happy"

        return "neutral"


def majority_vote(labels: Deque[str]) -> str:
    if not labels:
        return "none"
    counter = collections.Counter(labels)
    return counter.most_common(1)[0][0]


# Display text for each calibration expression (centered on screen)
_CALIBRATION_DISPLAY: Dict[str, str] = {
    "neutral": "Straight face",
    "happy": "Smile",
    "angry": "Frown",
}


def _draw_centered_text(
    frame: np.ndarray,
    text: str,
    y: int,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.7,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> None:
    """Draw a single line of text centered horizontally on the frame."""
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    w = frame.shape[1]
    x = (w - tw) // 2
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def _collect_calibration_samples(
    cap: cv2.VideoCapture,
    face_cascade: cv2.CascadeClassifier,
    predictor: dlib.shape_predictor,
    expression_name: str,
    step_number: int,
    total_steps: int,
) -> Optional[Tuple[float, float, float, float, float]]:
    """Collect 5 samples for a specific expression. User presses SPACE to start; 5 frames are captured automatically."""
    display_label = _CALIBRATION_DISPLAY.get(expression_name.lower(), expression_name)
    print(f"\n=== {expression_name.upper()} CALIBRATION (Step {step_number}/{total_steps}) ===")
    print(f"Show a {display_label.lower()}, then press SPACE or ENTER to capture 5 frames.")
    print("Press 'q' to cancel calibration.")
    
    samples = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(fps / 10))  # Sample ~10 times per second
    frame_count = 0
    collecting = False
    
    # Color for different expressions
    color_map = {
        "neutral": (0, 255, 0),  # Green
        "happy": (0, 255, 255),  # Yellow
        "angry": (0, 0, 255),    # Red
    }
    color = color_map.get(expression_name.lower(), (255, 255, 255))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Failed to read frame during {expression_name} calibration.")
            return None
        
        frame = cv2.flip(frame, 1)  # Flip horizontally to match main view
        h_f, w_f = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )
        
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_roi_gray = gray[y : y + h, x : x + w]

            # Analyze geometric features using landmarks
            eyebrow_height, mouth_width, mouth_height, mouth_curvature, eye_openness = analyze_geometric_features(
                face_roi_gray, predictor
            )
            
            # Collect 5 samples automatically once user has pressed SPACE
            if collecting and frame_count % frame_interval == 0:
                samples.append((eyebrow_height, mouth_width, mouth_height, mouth_curvature, eye_openness))
                if len(samples) >= 5:
                    break
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Center text vertically in the frame
            line_height = 36
            block_start = (h_f // 2) - (line_height * 2)  # roughly center 3 lines
            if not collecting:
                _draw_centered_text(frame, f"Step {step_number}/{total_steps}", block_start, font_scale=0.9, color=color)
                _draw_centered_text(frame, display_label, block_start + line_height)
                _draw_centered_text(frame, "Press SPACE or ENTER to capture 5 frames", block_start + line_height * 2, font_scale=0.6, color=(0, 255, 255))
            else:
                _draw_centered_text(frame, f"Capturing {display_label}...", block_start, font_scale=0.9, color=color)
                _draw_centered_text(frame, f"Frames: {len(samples)} / 5", block_start + line_height, font_scale=0.7, color=(0, 255, 255))
        
        cv2.imshow("Calibration", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print(f"Calibration cancelled.")
            cv2.destroyWindow("Calibration")
            return None
        elif (key == ord(" ") or key == 13) and not collecting:
            collecting = True
            samples = []
            frame_count = 0
            print(f"Capturing 5 frames for {expression_name}...")
        
        frame_count += 1
    
    cv2.destroyWindow("Calibration")
    
    if len(samples) < 5:
        print(f"Error: Not enough samples collected during {expression_name} calibration.")
        return None
    
    # Average the samples
    avg_eyebrow_height = np.mean([s[0] for s in samples])
    avg_mouth_width = np.mean([s[1] for s in samples])
    avg_mouth_height = np.mean([s[2] for s in samples])
    avg_mouth_curvature = np.mean([s[3] for s in samples])
    avg_eye_openness = np.mean([s[4] for s in samples])
    
    return (avg_eyebrow_height, avg_mouth_width, avg_mouth_height, avg_mouth_curvature, avg_eye_openness)


def calibrate_baseline(
    cap: cv2.VideoCapture,
    face_cascade: cv2.CascadeClassifier,
    predictor: dlib.shape_predictor,
    baseline: Baseline,
) -> bool:
    """
    Calibrate baseline values for neutral, happy, and angry expressions.
    User will be prompted to make each expression and press SPACE/ENTER when ready.
    """
    print("\n" + "="*50)
    print("CALIBRATION MODE")
    print("="*50)
    print("For each expression:")
    print("  1. Show the expression (straight face, smile, or frown)")
    print("  2. Press SPACE or ENTER to capture 5 frames automatically")
    print("  5. Press 'q' at any time to cancel")
    print("="*50 + "\n")
    
    # Step 1: Neutral expression
    neutral_samples = _collect_calibration_samples(
        cap, face_cascade, predictor, "neutral", step_number=1, total_steps=3
    )
    if neutral_samples is None:
        return False
    
    baseline.update_neutral(*neutral_samples)
    print(f"\n✓ Neutral calibration complete!")
    print(f"  Eyebrow height: {neutral_samples[0]:.3f}")
    print(f"  Mouth width: {neutral_samples[1]:.3f}, height: {neutral_samples[2]:.3f}")
    print("\nGet ready for the next step...\n")
    
    # Small delay to let user read the message
    time.sleep(1)
    
    # Step 2: Happy expression
    happy_samples = _collect_calibration_samples(
        cap, face_cascade, predictor, "happy", step_number=2, total_steps=3
    )
    if happy_samples is None:
        return False
    
    baseline.update_happy(happy_samples[1], happy_samples[2])  # width and height
    print(f"\n✓ Happy calibration complete!")
    print(f"  Mouth width: {happy_samples[1]:.3f}, height: {happy_samples[2]:.3f}")
    print("\nGet ready for the next step...\n")
    
    time.sleep(1)
    
    # Step 3: Angry expression
    angry_samples = _collect_calibration_samples(
        cap, face_cascade, predictor, "angry", step_number=3, total_steps=3
    )
    if angry_samples is None:
        return False
    
    baseline.update_angry(angry_samples[0])  # eyebrow height
    print(f"\n✓ Angry calibration complete!")
    print(f"  Eyebrow height: {angry_samples[0]:.3f}")
    
    baseline.finalize_calibration()
    
    print("\n" + "="*50)
    print("=== ALL CALIBRATIONS COMPLETE ===")
    print("="*50)
    print("You can now try different expressions!\n")
    
    return True


def debug_cameras() -> None:
    """
    Try each camera index and report whether it opens and whether frames can be read.
    Run with: python main.py --debug-camera
    """
    print("Camera diagnostic: trying indices 0–4 (OpenCV default backend).\n")
    for index in range(5):
        cap = cv2.VideoCapture(index)
        opened = cap.isOpened()
        if not opened:
            print(f"  Camera {index}: did not open")
            continue
        # Try to read a few frames; some devices open but never return frames
        read_ok = False
        for attempt in range(20):
            ret, frame = cap.read()
            if ret and frame is not None:
                read_ok = True
                h, w = frame.shape[:2]
                print(f"  Camera {index}: opened, reads frames OK (size {w}x{h})")
                break
        if not read_ok:
            print(f"  Camera {index}: opened but failed to read any frame (e.g. Continuity Camera on macOS)")
        cap.release()
    print("\nIf no camera 'reads frames OK', try:")
    print("  - macOS: disconnect Continuity Camera or pick built-in camera in System Settings > Camera")
    print("  - Grant camera permission to Terminal/IDE in System Settings > Privacy & Security > Camera")
    print("  - Close other apps using the camera.")


def _show_webcam_error_window() -> None:
    """Show a window with webcam error message so the user sees something instead of no window."""
    width, height = 640, 320
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)
    lines = [
        "Webcam could not deliver any frames.",
        "",
        "Another app may be using the camera (Zoom, FaceTime,",
        "Photo Booth, browser, etc.). Close it and try again.",
        "",
        "On macOS: disconnect Continuity Camera (iPhone)",
        "or choose built-in camera in System Settings > Camera.",
        "",
        "Check camera permission for Terminal/IDE in",
        "System Settings > Privacy & Security > Camera.",
        "",
        "Run:  python main.py --debug-camera  to diagnose.",
        "",
        "Press any key to close.",
    ]
    y = 28
    for line in lines:
        cv2.putText(img, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    cv2.imshow("Facial Emotion Demo", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Global variable for mouse callback
_start_button_clicked = False


def _welcome_mouse_callback(event, x, y, flags, param):
    """Mouse callback for welcome screen start button."""
    global _start_button_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # Button coordinates: (center_x - 100, center_y + 100) to (center_x + 100, center_y + 150)
        button_x1, button_y1 = param['button_x1'], param['button_y1']
        button_x2, button_y2 = param['button_x2'], param['button_y2']
        if button_x1 <= x <= button_x2 and button_y1 <= y <= button_y2:
            _start_button_clicked = True


def show_welcome_screen() -> bool:
    """
    Display welcome screen with start button.
    Returns True if start button was clicked, False if user closed window.
    """
    global _start_button_clicked
    _start_button_clicked = False
    
    width, height = 800, 600
    window_name = "Plant a Flower - Welcome"
    
    # Button dimensions
    button_width = 200
    button_height = 60
    button_x1 = (width - button_width) // 2
    button_y1 = height - 120  # Moved lower to avoid overlap
    button_x2 = button_x1 + button_width
    button_y2 = button_y1 + button_height
    
    # Set up mouse callback
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(
        window_name,
        _welcome_mouse_callback,
        {
            'button_x1': button_x1,
            'button_y1': button_y1,
            'button_x2': button_x2,
            'button_y2': button_y2,
        }
    )
    
    while True:
        # Create welcome screen image
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (40, 40, 40)  # Dark gray background
        
        # Welcome message
        message_lines = [
            "Hello and welcome to Plant a Flower!",
            "",
            "By clicking start, you consent to the",
            "use of facial recognition software.",
            "",
            "Help us grow this garden<3",
        ]
        
        y_offset = 120
        for i, line in enumerate(message_lines):
            font_scale = 0.7 if i == 0 else 0.55
            thickness = 2 if i == 0 else 1
            # Use FONT_HERSHEY_DUPLEX for a cleaner, more professional look
            font = cv2.FONT_HERSHEY_DUPLEX
            text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
            x_pos = (width - text_size[0]) // 2
            cv2.putText(
                img,
                line,
                (x_pos, y_offset + i * 45),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
        
        # Draw start button
        button_color = (0, 200, 0) if not _start_button_clicked else (0, 150, 0)
        cv2.rectangle(
            img,
            (button_x1, button_y1),
            (button_x2, button_y2),
            button_color,
            -1,
        )
        cv2.rectangle(
            img,
            (button_x1, button_y1),
            (button_x2, button_y2),
            (255, 255, 255),
            2,
        )
        
        # Button text
        button_text = "START"
        font = cv2.FONT_HERSHEY_DUPLEX  # Use cleaner font
        text_size = cv2.getTextSize(button_text, font, 0.9, 2)[0]
        text_x = button_x1 + (button_width - text_size[0]) // 2
        text_y = button_y1 + (button_height + text_size[1]) // 2
        cv2.putText(
            img,
            button_text,
            (text_x, text_y),
            font,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        
        cv2.imshow(window_name, img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            cv2.destroyWindow(window_name)
            return False
        
        if _start_button_clicked:
            cv2.destroyWindow(window_name)
            return True


def get_user_name() -> Optional[str]:
    """
    Display name input screen and collect user name via keyboard.
    Returns the name string or None if cancelled.
    """
    width, height = 600, 300
    window_name = "Enter Your Name"
    
    user_name = ""
    
    cv2.namedWindow(window_name)
    
    while True:
        # Create input screen image
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (40, 40, 40)  # Dark gray background
        
        # Instructions
        instruction = "Enter your name:"
        font = cv2.FONT_HERSHEY_DUPLEX  # Use cleaner font
        text_size = cv2.getTextSize(instruction, font, 0.65, 2)[0]
        x_pos = (width - text_size[0]) // 2
        cv2.putText(
            img,
            instruction,
            (x_pos, 80),
            font,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        
        # Display entered name
        display_text = user_name + "_" if len(user_name) > 0 else "_"
        name_size = cv2.getTextSize(display_text, font, 0.9, 2)[0]
        name_x = (width - name_size[0]) // 2
        cv2.putText(
            img,
            display_text,
            (name_x, 150),
            font,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        
        # Instructions
        hint = "Press ENTER to confirm, ESC to cancel, Backspace to delete"
        hint_size = cv2.getTextSize(hint, font, 0.5, 1)[0]
        hint_x = (width - hint_size[0]) // 2
        cv2.putText(
            img,
            hint,
            (hint_x, 220),
            font,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        
        cv2.imshow(window_name, img)
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == 27:  # ESC key
            cv2.destroyWindow(window_name)
            return None
        elif key == 13:  # ENTER key
            if len(user_name.strip()) > 0:
                cv2.destroyWindow(window_name)
                return user_name.strip()
            # If name is empty, continue
        elif key == 8 or key == 127:  # Backspace or Delete
            if len(user_name) > 0:
                user_name = user_name[:-1]
        elif 32 <= key <= 126:  # Printable ASCII characters
            if len(user_name) < 30:  # Limit name length
                user_name += chr(key)


def save_color_strip(history_strip: np.ndarray, user_name: str) -> None:
    """
    Stop on a review screen (legend + final strip), then save on confirmation.
    """
    if history_strip is None or history_strip.size == 0:
        print("No color strip to save.")
        return
    
    # Build a review screen (legend + final strip) that stays up until the user decides.
    legend_width, legend_height = 500, 400
    review_window = "Review - Legend & Final Color Strip"
    cv2.namedWindow(review_window, cv2.WINDOW_NORMAL)
    
    # Create legend image
    legend_img = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
    legend_img[:] = (40, 40, 40)
    
    # Legend title
    title = "Color Meanings:"
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    title_x = (legend_width - title_size[0]) // 2
    cv2.putText(
        legend_img,
        title,
        (title_x, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    
    # Legend items
    legend_items = [
        ("Dark gray", EMOTION_COLORS["none"], "No face detected"),
        ("Sage blossom", EMOTION_COLORS["neutral"], "Neutral"),
        ("Peony peach", EMOTION_COLORS["happy"], "Happy"),
        ("Burgundy rose", EMOTION_COLORS["angry"], "Angry"),
    ]
    
    y_start = 120
    for i, (color_name, bgr_color, emotion) in enumerate(legend_items):
        y_pos = y_start + i * 70
        
        # Color swatch
        cv2.rectangle(
            legend_img,
            (50, y_pos - 20),
            (120, y_pos + 20),
            bgr_color,
            -1,
        )
        cv2.rectangle(
            legend_img,
            (50, y_pos - 20),
            (120, y_pos + 20),
            (255, 255, 255),
            1,
        )
        
        # Color name and emotion
        cv2.putText(
            legend_img,
            f"{color_name} - {emotion}",
            (140, y_pos + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Create a larger preview of the final color strip for visibility
    strip_h, strip_w = history_strip.shape[:2]
    if strip_w <= 0:
        cv2.destroyWindow(review_window)
        print("No color strip to save.")
        return
    preview_w = legend_width
    preview_h = max(120, int(round(strip_h * (preview_w / float(strip_w)) * 6.0)))
    strip_preview = cv2.resize(history_strip, (preview_w, preview_h), interpolation=cv2.INTER_NEAREST)

    footer_h = 70
    canvas_h = legend_height + 12 + preview_h + footer_h
    canvas_w = legend_width
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)

    canvas[0:legend_height, 0:legend_width] = legend_img
    canvas[legend_height + 12 : legend_height + 12 + preview_h, 0:legend_width] = strip_preview

    instructions = "Press 's' (or ENTER) to SAVE | Press 'q' (or ESC) to cancel"
    cv2.putText(
        canvas,
        instructions,
        (10, canvas_h - 25),
        cv2.FONT_HERSHEY_DUPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Wait for explicit user confirmation before saving
    while True:
        cv2.imshow(review_window, canvas)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("s"), 13):  # 's' or ENTER
            break
        if key in (ord("q"), 27):  # 'q' or ESC
            cv2.destroyWindow(review_window)
            print("Save cancelled.")
            return

    cv2.destroyWindow(review_window)

    # Save color strip only (name is in the filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c for c in user_name if c.isalnum() or c in (' ', '-', '_')).strip()
    filename = f"{safe_name}_color_strip_{timestamp}.png"
    cv2.imwrite(filename, history_strip)
    print(f"\nColor strip saved as: {filename}")
    print(f"Saved in: {os.path.abspath(filename)}")


def _open_webcam() -> Optional[cv2.VideoCapture]:
    """
    Try to open a working webcam. On macOS, index 0 may be a Continuity Camera
    or external device that opens but fails to deliver frames; we try indices 0, 1, 2.
    """
    for index in (0, 1, 2):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            continue
        # Verify we can actually read a frame (some devices open but don't deliver frames)
        for _ in range(15):
            ret, _ = cap.read()
            if ret:
                return cap
        cap.release()
    return None


def main():
    # Show welcome screen
    if not show_welcome_screen():
        print("Welcome screen closed. Exiting.")
        return
    
    # Get user name
    user_name = get_user_name()
    if user_name is None:
        print("Name input cancelled. Exiting.")
        return
    
    print(f"Welcome, {user_name}!")
    
    # Open webcam
    cap = _open_webcam()

    if cap is None:
        print("Error: Could not open webcam.")
        print("  - On macOS: if you use Continuity Camera (iPhone), try disconnecting it")
        print("    or ensure the built-in camera is selected in System Settings.")
        print("  - Check that no other app is exclusively using the camera.")
        print("  - Grant camera permission to Terminal (or your IDE) in System Settings > Privacy.")
        return

    # Load Haar cascade for face detection
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        print(f"Error: Could not load face cascade from {face_cascade_path}")
        cap.release()
        return

    # Load dlib facial landmark predictor (68-point model)
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    try:
        predictor = dlib.shape_predictor(predictor_path)
    except RuntimeError:
        print(f"Error: Could not load dlib shape predictor from '{predictor_path}'.")
        print("Make sure the file exists in the project directory.")
        cap.release()
        return

    # History of recent emotion labels for smoothing
    recent_emotions: Deque[str] = collections.deque(maxlen=10)
    
    # Baseline for calibration
    baseline = Baseline()
    
    # Debug mode toggle
    debug_mode = False

    # Automatically start calibration
    print("\nStarting calibration...")
    if not calibrate_baseline(cap, face_cascade, predictor, baseline):
        print("Calibration failed or was cancelled. Exiting.")
        cap.release()
        cv2.destroyAllWindows()
        return

    print("\nPress 's' to stop (review legend + final strip, then save), 'd' to toggle debug mode, 'q' to quit without saving.")

    # Parameters for emotion history strip
    history_height = 40
    history_strip = None  # lazily initialized when we know frame width

    consecutive_failures = 0
    max_failures = 30  # ~1 second at 30 fps before giving up

    while True:
        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures == 1:
                print("Warning: Failed to read frame from webcam (retrying...).")
            if consecutive_failures >= max_failures:
                print("\nError: Webcam stopped returning frames.")
                print("  - On macOS with Continuity Camera: try disconnecting the iPhone or")
                print("    use a different camera in System Settings > Camera.")
                print("  - Close other apps using the camera and run this script again.")
                # Show an error window so something pops up (the main window never appears
                # because it is only drawn after we get a valid frame).
                _show_webcam_error_window()
                break
            continue
        consecutive_failures = 0
        frame = cv2.flip(frame, 1)

        # Initialize history strip once we know frame size
        if history_strip is None:
            h, w, _ = frame.shape
            history_strip = np.full((history_height, w, 3), EMOTION_COLORS["none"], dtype=np.uint8)

        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        current_emotion = "none"

        # Draw bounding box around the largest detected face (if any)
        if len(faces) > 0:
            # Choose the face with the largest area
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extract face ROI for emotion estimation
            face_roi_gray = gray[y : y + h, x : x + w]
            current_emotion = estimate_emotion_from_face_region(
                face_roi_gray, predictor=predictor, baseline=baseline, debug=debug_mode
            )

            # Draw debug info on frame if enabled
            if debug_mode:
                eyebrow_height, mouth_width, mouth_height, mouth_curvature, eye_openness = analyze_geometric_features(
                    face_roi_gray, predictor
                )

                debug_text = [
                    f"Eyebrow height: {eyebrow_height:.3f}",
                    f"Mouth: w={mouth_width:.3f}, h={mouth_height:.3f}",
                ]
                if baseline and baseline.calibrated:
                    debug_text.append(f"N: brow={baseline.neutral_eyebrow_height:.2f}, w={baseline.neutral_mouth_width:.2f}")
                    if baseline.happy_mouth_width:
                        debug_text.append(f"H: w={baseline.happy_mouth_width:.2f}, h={baseline.happy_mouth_height:.2f}")
                    if baseline.angry_eyebrow_height:
                        debug_text.append(f"A: brow={baseline.angry_eyebrow_height:.2f}")

                for i, text in enumerate(debug_text):
                    cv2.putText(
                        frame,
                        text,
                        (10, 60 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

        recent_emotions.append(current_emotion)
        smoothed_emotion = majority_vote(recent_emotions)
        color = EMOTION_COLORS.get(smoothed_emotion, EMOTION_COLORS["none"])

        # Only update history strip if calibration is complete
        if baseline.calibrated:
            # Shift history strip left by 1 pixel and add new color column at the right
            history_strip[:, 0:-1] = history_strip[:, 1:]
            history_strip[:, -1] = color
        else:
            # Keep history strip as "none" (dark gray) until calibrated
            history_strip[:, 0:-1] = history_strip[:, 1:]
            history_strip[:, -1] = EMOTION_COLORS["none"]

        # Combine the camera frame with the emotion history strip
        frame_with_history = np.vstack((frame, history_strip))

        # Draw instructions on the interface
        instructions = "Press 's' to stop (review legend + final strip, then save), 'd' to toggle debug mode, 'q' to quit without saving."
        cv2.putText(
            frame_with_history,
            instructions,
            (10, frame_with_history.shape[0] - 8),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("Facial Emotion Demo", frame_with_history)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):  # Stop and save
            save_color_strip(history_strip, user_name)
            break
        elif key == ord("q"):  # Quit without saving
            break
        elif key == ord("d"):
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Facial expression recognition from webcam.")
    parser.add_argument(
        "--debug-camera",
        action="store_true",
        help="Run camera diagnostic (which indices open and deliver frames), then exit.",
    )
    args = parser.parse_args()
    if args.debug_camera:
        debug_cameras()
        sys.exit(0)
    main()
