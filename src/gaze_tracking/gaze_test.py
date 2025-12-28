"""
Gaze Tracking Test - Iris based gaze + 5-point calibration
SPACE: start calibration
Q: quit
"""

import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh

LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

LEFT_EYE_INNER, LEFT_EYE_OUTER = 133, 33
RIGHT_EYE_INNER, RIGHT_EYE_OUTER = 362, 263

LEFT_EYE_TOP, LEFT_EYE_BOTTOM = 159, 145
RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM = 386, 374


def get_iris_center(landmarks, iris_indices, w, h):
    pts = [[landmarks[i].x * w, landmarks[i].y * h] for i in iris_indices]
    return np.mean(pts, axis=0)


def get_gaze_ratio_raw(landmarks, w, h):
    """
    Raw gaze signals:
      gaze_x around ~0..1 (but may be mirrored depending on definition)
      gaze_y around ~negative..positive
    We'll calibrate it anyway.
    """
    left_iris = get_iris_center(landmarks, LEFT_IRIS, w, h)
    right_iris = get_iris_center(landmarks, RIGHT_IRIS, w, h)

    l_inner = np.array([landmarks[LEFT_EYE_INNER].x * w, landmarks[LEFT_EYE_INNER].y * h])
    l_outer = np.array([landmarks[LEFT_EYE_OUTER].x * w, landmarks[LEFT_EYE_OUTER].y * h])
    r_inner = np.array([landmarks[RIGHT_EYE_INNER].x * w, landmarks[RIGHT_EYE_INNER].y * h])
    r_outer = np.array([landmarks[RIGHT_EYE_OUTER].x * w, landmarks[RIGHT_EYE_OUTER].y * h])

    # Horizontal ratio: define 0 at outer corner, 1 at inner corner (works with calibration)
    l_width = np.linalg.norm(l_inner - l_outer)
    r_width = np.linalg.norm(r_inner - r_outer)

    l_ratio = (left_iris[0] - l_outer[0]) / l_width if l_width > 1e-6 else 0.5
    r_ratio = (right_iris[0] - r_outer[0]) / r_width if r_width > 1e-6 else 0.5
    gaze_x = (l_ratio + r_ratio) / 2.0

    # Vertical: iris relative to eye center, normalized by eye height
    l_top = np.array([landmarks[LEFT_EYE_TOP].x * w, landmarks[LEFT_EYE_TOP].y * h])
    l_bot = np.array([landmarks[LEFT_EYE_BOTTOM].x * w, landmarks[LEFT_EYE_BOTTOM].y * h])
    r_top = np.array([landmarks[RIGHT_EYE_TOP].x * w, landmarks[RIGHT_EYE_TOP].y * h])
    r_bot = np.array([landmarks[RIGHT_EYE_BOTTOM].x * w, landmarks[RIGHT_EYE_BOTTOM].y * h])

    l_h = np.linalg.norm(l_top - l_bot)
    r_h = np.linalg.norm(r_top - r_bot)

    l_cy = (l_top[1] + l_bot[1]) / 2.0
    r_cy = (r_top[1] + r_bot[1]) / 2.0

    l_gy = (left_iris[1] - l_cy) / l_h if l_h > 1e-6 else 0.0
    r_gy = (right_iris[1] - r_cy) / r_h if r_h > 1e-6 else 0.0
    gaze_y = (l_gy + r_gy) / 2.0

    return float(gaze_x), float(gaze_y)


class Calibration5pt:
    def __init__(self):
        self.calibrated = False
        self.center = (0.5, 0.0)
        self.top = None
        self.bottom = None
        self.left = None
        self.right = None
        self.min_span = 1e-3

    def calibrate_5pt(self, center, top, bottom, left, right):
        self.center = center
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.calibrated = True
        print("✓ 5-point calibration saved:")
        print("  center:", self.center)
        print("  top   :", self.top)
        print("  bottom:", self.bottom)
        print("  left  :", self.left)
        print("  right :", self.right)

    def _span(self, a, b):
        return max(self.min_span, abs(a - b))

    def map_norm(self, gaze_x, gaze_y, sensitivity=1.0):
        """
        Returns normalized (nx, ny) in [0..1].
        nx=0 left, nx=1 right, ny=0 top, ny=1 bottom
        """
        if not self.calibrated:
            return 0.5, 0.5

        cx, cy = self.center

        # X: asymmetric spans
        if gaze_x >= cx:
            span = self._span(self.right[0], cx)
            dx = (gaze_x - cx) / span
        else:
            span = self._span(cx, self.left[0])
            dx = (gaze_x - cx) / span

        # Y: note "down" tends to increase gaze_y in our raw definition
        if gaze_y >= cy:
            span = self._span(self.bottom[1], cy)
            dy = (gaze_y - cy) / span
        else:
            span = self._span(cy, self.top[1])
            dy = (gaze_y - cy) / span

        dx *= sensitivity
        dy *= sensitivity

        # Map to 0..1
        nx = 0.5 + dx * 0.5
        ny = 0.5 - dy * 0.5  # dy positive => down

        nx = max(0.02, min(0.98, nx))
        ny = max(0.02, min(0.98, ny))
        return nx, ny


def main():
    cap = cv2.VideoCapture(0)

    calib = Calibration5pt()

    # Calibration flow
    calib_mode = False
    calib_stage = 0
    calib_samples = []
    samples_per_stage = 45

    collected = {"MERKEZ": None, "YUKARI": None, "ASAGI": None, "SOL": None, "SAG": None}

    def calib_targets(w, h):
        cx, cy = w // 2, h // 2
        margin = 120
        return [
            ("MERKEZ", (cx, cy)),
            ("YUKARI", (cx, margin)),
            ("ASAGI", (cx, h - margin)),
            ("SOL", (margin, cy)),
            ("SAG", (w - margin, cy)),
        ]

    # smoothing for visualization
    smooth_nx, smooth_ny = 0.5, 0.5
    smooth_alpha = 0.18
    
    # Set window size to 1920x1080
    window_name = "Gaze Tracking Test (Calibrated)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1920, 1080)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        print("SPACE: calibration | Q: quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            gaze_x, gaze_y = 0.5, 0.0

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                gaze_x, gaze_y = get_gaze_ratio_raw(landmarks, w, h)

                # collect samples if calibrating
                if calib_mode:
                    calib_samples.append((gaze_x, gaze_y))
                    if len(calib_samples) >= samples_per_stage:
                        label, _ = calib_targets(w, h)[calib_stage]
                        avg_x = float(np.mean([s[0] for s in calib_samples]))
                        avg_y = float(np.mean([s[1] for s in calib_samples]))
                        collected[label] = (avg_x, avg_y)

                        calib_samples = []
                        calib_stage += 1

                        if calib_stage >= 5:
                            calib.calibrate_5pt(
                                center=collected["MERKEZ"],
                                top=collected["YUKARI"],
                                bottom=collected["ASAGI"],
                                left=collected["SOL"],
                                right=collected["SAG"],
                            )
                            calib_mode = False
                            calib_stage = 0
                        else:
                            nxt, _ = calib_targets(w, h)[calib_stage]
                            print("→ look at:", nxt)

            # Use calibration to map to normalized screen point
            nx, ny = calib.map_norm(gaze_x, gaze_y, sensitivity=1.0) if calib.calibrated else (0.5, 0.5)

            # smooth for stable viz
            smooth_nx = smooth_nx + smooth_alpha * (nx - smooth_nx)
            smooth_ny = smooth_ny + smooth_alpha * (ny - smooth_ny)

            # Decide direction (based on calibrated nx/ny)
            direction = ""
            if smooth_nx < 0.4:
                direction = "LEFT"
            elif smooth_nx > 0.6:
                direction = "RIGHT"
            else:
                direction = "CENTER"

            if smooth_ny < 0.4:
                direction += " UP"
            elif smooth_ny > 0.6:
                direction += " DOWN"

            # HUD
            cv2.putText(frame, f"raw gaze_x: {gaze_x:.3f}  raw gaze_y: {gaze_y:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"calib nx: {smooth_nx:.2f}  ny: {smooth_ny:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"DIR: {direction}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # visualize point (box bottom-left)
            box_x1, box_y1 = 50, h - 180
            box_x2, box_y2 = 300, h - 50
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (50, 50, 50), -1)
            px = int(box_x1 + smooth_nx * (box_x2 - box_x1))
            py = int(box_y1 + smooth_ny * (box_y2 - box_y1))
            cv2.circle(frame, (px, py), 8, (0, 0, 255), -1)
            cv2.putText(frame, "Calibrated gaze point", (box_x1, box_y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # calibration overlay
            if calib_mode:
                label, pos = calib_targets(w, h)[calib_stage]
                cv2.rectangle(frame, (0, 0), (w, h), (0, 100, 100), 4)
                cv2.putText(frame, f"CALIB: {label} ({len(calib_samples)}/{samples_per_stage})",
                            (w // 2 - 220, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.circle(frame, pos, 28, (0, 255, 255), 3)
                cv2.circle(frame, pos, 6, (0, 255, 255), -1)

            cv2.putText(frame, "[SPACE]=calib  [Q]=quit", (w - 260, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                calib_mode = True
                calib_stage = 0
                calib_samples = []
                for k in collected:
                    collected[k] = None
                print("Calibration started → look at: MERKEZ")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
