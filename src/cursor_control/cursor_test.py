"""
Cursor Control Test - Kalibrasyonlu iris tabanlı cursor kontrolü
Düzgün kalibrasyon ile ekran koordinatlarına haritalama yapar.
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh

# İris landmark indeksleri
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Göz köşeleri
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263

# Göz üst/alt
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374


class GazeCalibration:
    """Bakış kalibrasyonu için sınıf"""
    
    def __init__(self):
        self.calibrated = False
        
        # Kalibrasyon noktaları (bakış oranları)
        self.center_gaze = None  # Merkeze bakarken
        self.left_gaze = None    # Sola bakarken
        self.right_gaze = None   # Sağa bakarken
        self.top_gaze = None     # Yukarı bakarken
        self.bottom_gaze = None  # Aşağı bakarken
        
        # Hesaplanan aralıklar
        self.gaze_x_min = 0.3
        self.gaze_x_max = 0.7
        self.gaze_y_min = -0.5
        self.gaze_y_max = 0.5
        
    def quick_calibrate(self, center_x, center_y):
        """Hızlı kalibrasyon - sadece merkez noktası"""
        self.center_gaze = (center_x, center_y)
        
        # Varsayılan aralıklar (merkeze göre ayarla)
        self.gaze_x_min = center_x - 0.25
        self.gaze_x_max = center_x + 0.25
        self.gaze_y_min = center_y - 0.4
        self.gaze_y_max = center_y + 0.4
        
        self.calibrated = True
        
    def map_to_screen(self, gaze_x, gaze_y, screen_w, screen_h):
        """Bakış oranını ekran koordinatına çevir"""
        if not self.calibrated:
            # Kalibrasyon yoksa varsayılan haritalama
            norm_x = (gaze_x - 0.3) / 0.4  # 0.3-0.7 aralığını 0-1'e
            norm_y = (gaze_y + 0.5) / 1.0  # -0.5-0.5 aralığını 0-1'e
        else:
            # Kalibre edilmiş haritalama
            norm_x = (gaze_x - self.gaze_x_min) / (self.gaze_x_max - self.gaze_x_min)
            norm_y = (gaze_y - self.gaze_y_min) / (self.gaze_y_max - self.gaze_y_min)
        
        # Sınırla
        norm_x = max(0, min(1, norm_x))
        norm_y = max(0, min(1, norm_y))
        
        # Ekran koordinatına çevir (X ters çünkü ayna efekti)
        screen_x = int((1 - norm_x) * screen_w)
        screen_y = int(norm_y * screen_h)
        
        return screen_x, screen_y


def get_iris_center(landmarks, iris_indices, w, h):
    """İris merkezini hesapla"""
    points = []
    for idx in iris_indices:
        lm = landmarks[idx]
        points.append([lm.x * w, lm.y * h])
    return np.mean(points, axis=0)


def get_gaze_ratio(landmarks, w, h):
    """Bakış oranını hesapla"""
    # İris merkezleri
    left_iris = get_iris_center(landmarks, LEFT_IRIS, w, h)
    right_iris = get_iris_center(landmarks, RIGHT_IRIS, w, h)
    
    # Göz köşeleri
    left_inner = np.array([landmarks[LEFT_EYE_INNER].x * w, landmarks[LEFT_EYE_INNER].y * h])
    left_outer = np.array([landmarks[LEFT_EYE_OUTER].x * w, landmarks[LEFT_EYE_OUTER].y * h])
    right_inner = np.array([landmarks[RIGHT_EYE_INNER].x * w, landmarks[RIGHT_EYE_INNER].y * h])
    right_outer = np.array([landmarks[RIGHT_EYE_OUTER].x * w, landmarks[RIGHT_EYE_OUTER].y * h])
    
    # Yatay oran
    left_eye_width = np.linalg.norm(left_inner - left_outer)
    right_eye_width = np.linalg.norm(right_inner - right_outer)
    
    left_ratio = (left_iris[0] - left_outer[0]) / left_eye_width if left_eye_width > 0 else 0.5
    right_ratio = (right_iris[0] - right_outer[0]) / right_eye_width if right_eye_width > 0 else 0.5
    
    gaze_x = (left_ratio + right_ratio) / 2
    
    # Dikey oran
    left_top = np.array([landmarks[LEFT_EYE_TOP].x * w, landmarks[LEFT_EYE_TOP].y * h])
    left_bottom = np.array([landmarks[LEFT_EYE_BOTTOM].x * w, landmarks[LEFT_EYE_BOTTOM].y * h])
    right_top = np.array([landmarks[RIGHT_EYE_TOP].x * w, landmarks[RIGHT_EYE_TOP].y * h])
    right_bottom = np.array([landmarks[RIGHT_EYE_BOTTOM].x * w, landmarks[RIGHT_EYE_BOTTOM].y * h])
    
    left_eye_height = np.linalg.norm(left_top - left_bottom)
    right_eye_height = np.linalg.norm(right_top - right_bottom)
    
    left_center_y = (left_top[1] + left_bottom[1]) / 2
    right_center_y = (right_top[1] + right_bottom[1]) / 2
    
    left_gaze_y = (left_iris[1] - left_center_y) / (left_eye_height) if left_eye_height > 0 else 0
    right_gaze_y = (right_iris[1] - right_center_y) / (right_eye_height) if right_eye_height > 0 else 0
    
    gaze_y = (left_gaze_y + right_gaze_y) / 2
    
    return gaze_x, gaze_y


def main():
    cap = cv2.VideoCapture(0)
    pyautogui.FAILSAFE = False
    
    screen_w, screen_h = pyautogui.size()
    
    # Kalibrasyon
    calibration = GazeCalibration()
    
    # Smoothing için
    SMOOTHING = 0.12  # Daha düşük = daha yumuşak ama yavaş
    prev_x, prev_y = screen_w // 2, screen_h // 2
    
    # Cursor kontrolü açık/kapalı
    cursor_enabled = False
    
    # Kalibrasyon modu
    calibration_mode = False
    calibration_samples = []
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        print("\n=== BlinkTrack Cursor Control ===")
        print("Kontroller:")
        print("  C - Kalibrasyon (ekran merkezine bak ve bas)")
        print("  SPACE - Cursor kontrolünü aç/kapa")
        print("  Q - Çıkış")
        print("  +/- - Hassasiyet ayarla")
        print()
        
        # Hassasiyet çarpanı
        sensitivity = 1.0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            
            gaze_x, gaze_y = 0.5, 0
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Bakış oranını hesapla
                gaze_x, gaze_y = get_gaze_ratio(landmarks, w, h)
                
                # İris çiz
                left_iris = get_iris_center(landmarks, LEFT_IRIS, w, h)
                right_iris = get_iris_center(landmarks, RIGHT_IRIS, w, h)
                cv2.circle(frame, (int(left_iris[0]), int(left_iris[1])), 3, (0, 255, 0), -1)
                cv2.circle(frame, (int(right_iris[0]), int(right_iris[1])), 3, (0, 255, 0), -1)
                
                # Kalibrasyon modu
                if calibration_mode:
                    calibration_samples.append((gaze_x, gaze_y))
                    cv2.putText(frame, f"Kalibrasyon... {len(calibration_samples)}/30", 
                               (w//2 - 100, h//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
                    if len(calibration_samples) >= 30:
                        # Ortalama al
                        avg_x = np.mean([s[0] for s in calibration_samples])
                        avg_y = np.mean([s[1] for s in calibration_samples])
                        calibration.quick_calibrate(avg_x, avg_y)
                        calibration_mode = False
                        calibration_samples = []
                        print(f"Kalibrasyon tamamlandı! Merkez: ({avg_x:.3f}, {avg_y:.3f})")
                
                # Cursor kontrolü
                if cursor_enabled and calibration.calibrated:
                    # Ekran koordinatına çevir
                    target_x, target_y = calibration.map_to_screen(
                        gaze_x, gaze_y, screen_w, screen_h
                    )
                    
                    # Hassasiyet uygula (merkezden uzaklığa göre)
                    center_x, center_y = screen_w // 2, screen_h // 2
                    target_x = int(center_x + (target_x - center_x) * sensitivity)
                    target_y = int(center_y + (target_y - center_y) * sensitivity)
                    
                    # Smoothing
                    smooth_x = int(prev_x + (target_x - prev_x) * SMOOTHING)
                    smooth_y = int(prev_y + (target_y - prev_y) * SMOOTHING)
                    
                    # Sınırları kontrol et
                    smooth_x = max(0, min(screen_w - 1, smooth_x))
                    smooth_y = max(0, min(screen_h - 1, smooth_y))
                    
                    prev_x, prev_y = smooth_x, smooth_y
                    
                    # Cursor'ı hareket ettir
                    pyautogui.moveTo(smooth_x, smooth_y, duration=0)
            
            # Durum bilgisi
            status_color = (0, 255, 0) if cursor_enabled else (0, 0, 255)
            status_text = "CURSOR: ACIK" if cursor_enabled else "CURSOR: KAPALI"
            cv2.putText(frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            calib_text = "Kalibre" if calibration.calibrated else "Kalibre Degil"
            calib_color = (0, 255, 0) if calibration.calibrated else (0, 165, 255)
            cv2.putText(frame, calib_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, calib_color, 2)
            
            cv2.putText(frame, f"Gaze: ({gaze_x:.2f}, {gaze_y:.2f})", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.putText(frame, f"Hassasiyet: {sensitivity:.1f}x", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Kontrol bilgisi
            cv2.putText(frame, "C: Kalibrasyon | SPACE: Cursor | +/-: Hassasiyet | Q: Cikis", 
                       (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Kalibrasyon hedefi (merkez)
            if not calibration.calibrated or calibration_mode:
                cv2.circle(frame, (w//2, h//2), 20, (0, 255, 255), 2)
                cv2.circle(frame, (w//2, h//2), 5, (0, 255, 255), -1)
                cv2.putText(frame, "Merkeze bak ve C bas", (w//2 - 100, h//2 + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow("Cursor Control", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                calibration_mode = True
                calibration_samples = []
                print("Kalibrasyon başladı - Ekran merkezine bakın...")
            elif key == ord(' '):
                if calibration.calibrated:
                    cursor_enabled = not cursor_enabled
                    print(f"Cursor: {'Açık' if cursor_enabled else 'Kapalı'}")
                else:
                    print("Önce kalibrasyon yapın! (C tuşu)")
            elif key == ord('+') or key == ord('='):
                sensitivity = min(3.0, sensitivity + 0.2)
                print(f"Hassasiyet: {sensitivity:.1f}x")
            elif key == ord('-'):
                sensitivity = max(0.4, sensitivity - 0.2)
                print(f"Hassasiyet: {sensitivity:.1f}x")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()