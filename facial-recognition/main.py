import collections
import time
from typing import Deque, Dict, Optional, Tuple

import cv2
import dlib
import numpy as np


EMOTION_COLORS: Dict[str, tuple] = {
    "none": (50, 50, 50),       # dark gray
    "neutral": (255, 0, 0),     # blue (BGR)
    "happy": (0, 255, 255),     # yellow (BGR)
    "angry": (0, 0, 255),       # red (BGR)
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


def _collect_calibration_samples(
    cap: cv2.VideoCapture,
    face_cascade: cv2.CascadeClassifier,
    predictor: dlib.shape_predictor,
    expression_name: str,
    step_number: int,
    total_steps: int,
) -> Optional[Tuple[float, float, float, float, float]]:
    """Collect samples for a specific expression. User presses SPACE or ENTER when ready."""
    print(f"\n=== {expression_name.upper()} CALIBRATION (Step {step_number}/{total_steps}) ===")
    print(f"Make a {expression_name.upper()} expression, then press SPACE or ENTER when ready to capture.")
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
            
            # Collect samples while collecting flag is True
            if collecting and frame_count % frame_interval == 0:
                samples.append((eyebrow_height, mouth_width, mouth_height, mouth_curvature, eye_openness))
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Show different messages based on state
            if not collecting:
                cv2.putText(
                    frame,
                    f"Step {step_number}/{total_steps}: {expression_name.upper()}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Make a {expression_name.upper()} expression",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    "Press SPACE or ENTER to start capturing",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    frame,
                    f"Capturing {expression_name.upper()}...",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Keep the {expression_name.upper()} expression",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Samples: {len(samples)} | Press SPACE/ENTER when done",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
        
        cv2.imshow("Calibration", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print(f"Calibration cancelled.")
            cv2.destroyWindow("Calibration")
            return None
        elif key == ord(" ") or key == 13:  # SPACE or ENTER
            if not collecting:
                # Start collecting samples
                collecting = True
                samples = []  # Reset samples
                frame_count = 0
                print(f"Collecting samples for {expression_name}...")
            else:
                # Stop collecting and finish
                if len(samples) >= 5:
                    break
                else:
                    print(f"Need at least 5 samples. Currently have {len(samples)}. Keep collecting...")
        
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
    print("  1. Make the expression")
    print("  2. Press SPACE or ENTER to start capturing")
    print("  3. Hold the expression")
    print("  4. Press SPACE or ENTER again when done")
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


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Load Haar cascade for face detection
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        print(f"Error: Could not load face cascade from {face_cascade_path}")
        return

    # Load dlib facial landmark predictor (68-point model)
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    try:
        predictor = dlib.shape_predictor(predictor_path)
    except RuntimeError:
        print(f"Error: Could not load dlib shape predictor from '{predictor_path}'.")
        print("Make sure the file exists in the project directory.")
        return

    # History of recent emotion labels for smoothing
    recent_emotions: Deque[str] = collections.deque(maxlen=10)
    
    # Baseline for calibration
    baseline = Baseline()
    
    # Debug mode toggle
    debug_mode = False

    print("Press 'q' to quit, 'c' to calibrate, 'd' to toggle debug mode.")
    print("IMPORTANT: Press 'c' first to calibrate with a neutral expression!")
    print("Then try: smiling (yellow), furrowing eyebrows (red), or neutral (blue)")

    # Parameters for emotion history strip
    history_height = 40
    history_strip = None  # lazily initialized when we know frame width

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to read frame from webcam.")
            break

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

        # Draw current emotion text on the frame
        cv2.putText(
            frame,
            f"Emotion: {smoothed_emotion}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
        
        # Show calibration status
        if not baseline.calibrated:
            cv2.putText(
                frame,
                "NOT CALIBRATED - Press 'c'",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # Combine the camera frame with the emotion history strip
        frame_with_history = np.vstack((frame, history_strip))

        cv2.imshow("Facial Emotion Demo", frame_with_history)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            calibrate_baseline(cap, face_cascade, predictor, baseline)
        elif key == ord("d"):
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
