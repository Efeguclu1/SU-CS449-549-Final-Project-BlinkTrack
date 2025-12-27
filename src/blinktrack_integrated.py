"""
BlinkTrack - Entegre Sistem
Bakış ile cursor kontrolü + Çift blink ile tıklama
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

mp_face_mesh = mp.solutions.face_mesh

# === LANDMARK İNDEKSLERİ ===

# İris
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Göz köşeleri
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263

# Göz üst/alt (EAR için)
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

# EAR hesaplama için
LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR = [263, 387, 385, 362, 380, 373]


# === YARDIMCI FONKSİYONLAR ===

def euclidean_dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def get_iris_center(landmarks, iris_indices, w, h):
    points = [[landmarks[i].x * w, landmarks[i].y * h] for i in iris_indices]
    return np.mean(points, axis=0)


def eye_aspect_ratio(eye_points):
    v1 = euclidean_dist(eye_points[1], eye_points[5])
    v2 = euclidean_dist(eye_points[2], eye_points[4])
    h = euclidean_dist(eye_points[0], eye_points[3])
    return (v1 + v2) / (2.0 * h) if h > 0 else 0


def get_gaze_ratio(landmarks, w, h):
    """Bakış oranını hesapla (0-1)"""
    left_iris = get_iris_center(landmarks, LEFT_IRIS, w, h)
    right_iris = get_iris_center(landmarks, RIGHT_IRIS, w, h)
    
    # Göz köşeleri
    l_inner = np.array([landmarks[LEFT_EYE_INNER].x * w, landmarks[LEFT_EYE_INNER].y * h])
    l_outer = np.array([landmarks[LEFT_EYE_OUTER].x * w, landmarks[LEFT_EYE_OUTER].y * h])
    r_inner = np.array([landmarks[RIGHT_EYE_INNER].x * w, landmarks[RIGHT_EYE_INNER].y * h])
    r_outer = np.array([landmarks[RIGHT_EYE_OUTER].x * w, landmarks[RIGHT_EYE_OUTER].y * h])
    
    # Yatay
    l_width = np.linalg.norm(l_inner - l_outer)
    r_width = np.linalg.norm(r_inner - r_outer)
    
    l_ratio = (left_iris[0] - l_outer[0]) / l_width if l_width > 0 else 0.5
    r_ratio = (right_iris[0] - r_outer[0]) / r_width if r_width > 0 else 0.5
    gaze_x = (l_ratio + r_ratio) / 2
    
    # Dikey
    l_top = np.array([landmarks[LEFT_EYE_TOP].x * w, landmarks[LEFT_EYE_TOP].y * h])
    l_bot = np.array([landmarks[LEFT_EYE_BOTTOM].x * w, landmarks[LEFT_EYE_BOTTOM].y * h])
    r_top = np.array([landmarks[RIGHT_EYE_TOP].x * w, landmarks[RIGHT_EYE_TOP].y * h])
    r_bot = np.array([landmarks[RIGHT_EYE_BOTTOM].x * w, landmarks[RIGHT_EYE_BOTTOM].y * h])
    
    l_height = np.linalg.norm(l_top - l_bot)
    r_height = np.linalg.norm(r_top - r_bot)
    
    l_center_y = (l_top[1] + l_bot[1]) / 2
    r_center_y = (r_top[1] + r_bot[1]) / 2
    
    l_gaze_y = (left_iris[1] - l_center_y) / l_height if l_height > 0 else 0
    r_gaze_y = (right_iris[1] - r_center_y) / r_height if r_height > 0 else 0
    gaze_y = (l_gaze_y + r_gaze_y) / 2
    
    return gaze_x, gaze_y


# === ANA SINIFLAR ===

class Calibration:
    def __init__(self):
        self.calibrated = False
        self.center_x = 0.5
        self.center_y = 0.0
        # Daha küçük range = daha hassas kontrol
        self.range_x = 0.12
        self.range_y = 0.20
        
    def calibrate(self, gaze_x, gaze_y):
        self.center_x = gaze_x
        self.center_y = gaze_y
        self.calibrated = True
        
    def map_to_screen(self, gaze_x, gaze_y, screen_w, screen_h, sensitivity=1.0):
        if not self.calibrated:
            return screen_w // 2, screen_h // 2
        
        # Merkeze göre fark
        diff_x = (gaze_x - self.center_x) / self.range_x
        diff_y = (gaze_y - self.center_y) / self.range_y
        
        # Hassasiyet uygula
        diff_x *= sensitivity
        diff_y *= sensitivity
        
        # Ekran koordinatına çevir
        norm_x = 0.5 - diff_x * 0.5
        norm_y = 0.5 + diff_y * 0.5
        
        # Sınırla
        norm_x = max(0, min(1, norm_x))
        norm_y = max(0, min(1, norm_y))
        
        return int(norm_x * screen_w), int(norm_y * screen_h)


class BlinkDetector:
    def __init__(self, threshold=0.24):  # 0.21'den 0.24'e yükseltildi
        self.threshold = threshold
        self.eye_closed = False
        self.close_time = None
        self.last_blink = None
        self.pending = False
        
        # Zamanlamalar (biraz gevşetildi)
        self.max_single = 0.4           # 0.35'ten 0.4'e
        self.double_window = 0.55       # 0.45'ten 0.55'e
        self.prolonged = 0.8            # 0.7'den 0.8'e
        
        # Debug için
        self.min_ear_seen = 1.0
        self.max_ear_seen = 0.0
        
    def update(self, ear):
        now = time.time()
        result = None
        
        # Min/max takibi (debug için)
        self.min_ear_seen = min(self.min_ear_seen, ear)
        self.max_ear_seen = max(self.max_ear_seen, ear)
        
        closed = ear < self.threshold
        
        if closed and not self.eye_closed:
            self.eye_closed = True
            self.close_time = now
            
        elif not closed and self.eye_closed:
            self.eye_closed = False
            
            if self.close_time:
                duration = now - self.close_time
                
                if duration >= self.prolonged:
                    result = "prolonged"
                    self.pending = False
                elif duration <= self.max_single:
                    if self.pending and self.last_blink and (now - self.last_blink) <= self.double_window:
                        result = "double"
                        self.pending = False
                    else:
                        self.pending = True
                        self.last_blink = now
            
            self.close_time = None
        
        # Tek blink zamanı geçti mi
        if self.pending and self.last_blink and (now - self.last_blink) > self.double_window:
            result = "single"
            self.pending = False
        
        return result
    
    def reset_ear_stats(self):
        """Min/max EAR istatistiklerini sıfırla"""
        self.min_ear_seen = 1.0
        self.max_ear_seen = 0.0


# === ANA FONKSİYON ===

def main():
    cap = cv2.VideoCapture(0)
    pyautogui.FAILSAFE = False
    
    screen_w, screen_h = pyautogui.size()
    
    calibration = Calibration()
    blink_detector = BlinkDetector(threshold=0.21)
    
    # Ayarlar
    cursor_enabled = False
    smoothing = 0.25       # Daha hızlı tepki
    sensitivity = 1.5      # Daha hassas
    prev_x, prev_y = screen_w // 2, screen_h // 2
    
    # Kalibrasyon modu
    calib_mode = False
    calib_samples = []
    
    # Görsel geri bildirim
    last_blink_type = None
    last_blink_time = 0
    click_count = 0
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        print("\n" + "="*50)
        print("       BlinkTrack - Entegre Sistem")
        print("="*50)
        print("\nKontroller:")
        print("  C     - Kalibrasyon (ekran merkezine bakın)")
        print("  SPACE - Cursor kontrolü aç/kapa")
        print("  +/-   - Hassasiyet ayarla")
        print("  T/G   - EAR eşik ayarla (T:artır, G:azalt)")
        print("  R     - Min/Max EAR sıfırla")
        print("  Q     - Çıkış")
        print("\nBlink Algılama İpucu:")
        print("  1. Göz kırpın ve Min EAR değerine bakın")
        print("  2. Eşik değeri Min EAR'dan yüksek olmalı")
        print("  3. T tuşu ile eşiği artırın")
        print("\nÇift blink = Tıklama!")
        print("="*50 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            
            ear = 0.3
            gaze_x, gaze_y = 0.5, 0.0
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Bakış oranı
                gaze_x, gaze_y = get_gaze_ratio(landmarks, w, h)
                
                # EAR hesapla
                left_pts = [[landmarks[i].x * w, landmarks[i].y * h] for i in LEFT_EYE_EAR]
                right_pts = [[landmarks[i].x * w, landmarks[i].y * h] for i in RIGHT_EYE_EAR]
                ear = (eye_aspect_ratio(left_pts) + eye_aspect_ratio(right_pts)) / 2
                
                # İrisleri çiz
                left_iris = get_iris_center(landmarks, LEFT_IRIS, w, h)
                right_iris = get_iris_center(landmarks, RIGHT_IRIS, w, h)
                cv2.circle(frame, (int(left_iris[0]), int(left_iris[1])), 4, (0, 255, 0), -1)
                cv2.circle(frame, (int(right_iris[0]), int(right_iris[1])), 4, (0, 255, 0), -1)
                
                # Kalibrasyon modu
                if calib_mode:
                    calib_samples.append((gaze_x, gaze_y))
                    cv2.putText(frame, f"Kalibre ediliyor... {len(calib_samples)}/30",
                               (w//2 - 120, h//2 + 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Hedef göster
                    cv2.circle(frame, (w//2, h//2), 30, (0, 255, 255), 3)
                    cv2.circle(frame, (w//2, h//2), 8, (0, 255, 255), -1)
                    
                    if len(calib_samples) >= 30:
                        avg_x = np.mean([s[0] for s in calib_samples])
                        avg_y = np.mean([s[1] for s in calib_samples])
                        calibration.calibrate(avg_x, avg_y)
                        calib_mode = False
                        calib_samples = []
                        print(f"✓ Kalibrasyon tamamlandı!")
                
                # Blink algılama
                blink = blink_detector.update(ear)
                if blink:
                    last_blink_type = blink
                    last_blink_time = time.time()
                    
                    if blink == "double" and cursor_enabled:
                        pyautogui.click()
                        click_count += 1
                        print(f"[CLICK!] Toplam: {click_count}")
                
                # Cursor kontrolü
                if cursor_enabled and calibration.calibrated:
                    target_x, target_y = calibration.map_to_screen(
                        gaze_x, gaze_y, screen_w, screen_h, sensitivity
                    )
                    
                    # Smoothing
                    smooth_x = int(prev_x + (target_x - prev_x) * smoothing)
                    smooth_y = int(prev_y + (target_y - prev_y) * smoothing)
                    
                    smooth_x = max(0, min(screen_w - 1, smooth_x))
                    smooth_y = max(0, min(screen_h - 1, smooth_y))
                    
                    prev_x, prev_y = smooth_x, smooth_y
                    pyautogui.moveTo(smooth_x, smooth_y, duration=0)
            
            # === UI GÖSTERGE ===
            
            # Durum paneli (biraz daha geniş)
            panel_h = 160
            cv2.rectangle(frame, (0, 0), (280, panel_h), (30, 30, 30), -1)
            
            # Cursor durumu
            status = "AKTIF" if cursor_enabled else "KAPALI"
            color = (0, 255, 0) if cursor_enabled else (0, 0, 255)
            cv2.putText(frame, f"Cursor: {status}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Kalibrasyon durumu
            calib_status = "OK" if calibration.calibrated else "Gerekli"
            calib_color = (0, 255, 0) if calibration.calibrated else (0, 165, 255)
            cv2.putText(frame, f"Kalibrasyon: {calib_status}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, calib_color, 2)
            
            # EAR (eşik ile birlikte)
            ear_color = (0, 255, 0) if ear >= blink_detector.threshold else (0, 0, 255)
            cv2.putText(frame, f"EAR: {ear:.3f} (esik: {blink_detector.threshold:.2f})", (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, ear_color, 2)
            
            # Min/Max EAR (eşik ayarlamak için yardımcı)
            cv2.putText(frame, f"Min: {blink_detector.min_ear_seen:.3f} Max: {blink_detector.max_ear_seen:.3f}", 
                       (10, 97),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 200, 255), 1)
            
            # Gaze
            cv2.putText(frame, f"Gaze: ({gaze_x:.2f}, {gaze_y:.2f})", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Hassasiyet ve click sayısı
            cv2.putText(frame, f"Hassasiyet: {sensitivity:.1f}x | Click: {click_count}", 
                       (10, 145),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Blink bildirimi
            if last_blink_type and (time.time() - last_blink_time) < 0.6:
                if last_blink_type == "double":
                    text = "CIFT BLINK - CLICK!"
                    bcolor = (0, 255, 0)
                elif last_blink_type == "prolonged":
                    text = "UZUN BLINK"
                    bcolor = (255, 0, 255)
                else:
                    text = "TEK BLINK"
                    bcolor = (255, 255, 0)
                
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = (w - text_size[0]) // 2
                cv2.putText(frame, text, (text_x, h - 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, bcolor, 2)
            
            # Kontrol yardımı
            cv2.putText(frame, "C:Kalibrasyon SPACE:Cursor +/-:Hassasiyet T/G:Esik R:Reset Q:Cikis",
                       (10, h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
            # Kalibrasyon gerekli uyarısı
            if not calibration.calibrated and not calib_mode:
                cv2.rectangle(frame, (w//2 - 180, h//2 - 60), (w//2 + 180, h//2 + 60), 
                             (0, 0, 100), -1)
                cv2.putText(frame, "Kalibrasyon Gerekli!", (w//2 - 130, h//2 - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, "Ekran merkezine bakin ve", (w//2 - 140, h//2 + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, "C tusuna basin", (w//2 - 80, h//2 + 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("BlinkTrack", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                calib_mode = True
                calib_samples = []
                print("Kalibrasyon başladı - Ekran merkezine bakın...")
            elif key == ord(' '):
                if calibration.calibrated:
                    cursor_enabled = not cursor_enabled
                    print(f"Cursor: {'Açık' if cursor_enabled else 'Kapalı'}")
                else:
                    print("Önce kalibrasyon yapın! (C tuşu)")
            elif key == ord('+') or key == ord('='):
                sensitivity = min(3.0, sensitivity + 0.1)
                print(f"Hassasiyet: {sensitivity:.1f}x")
            elif key == ord('-'):
                sensitivity = max(0.5, sensitivity - 0.1)
                print(f"Hassasiyet: {sensitivity:.1f}x")
            elif key == ord('t'):
                # Eşik artır (daha kolay blink algılama)
                blink_detector.threshold = min(0.35, blink_detector.threshold + 0.01)
                print(f"EAR Eşik: {blink_detector.threshold:.2f} (artırıldı)")
            elif key == ord('g'):
                # Eşik azalt (daha zor blink algılama)
                blink_detector.threshold = max(0.15, blink_detector.threshold - 0.01)
                print(f"EAR Eşik: {blink_detector.threshold:.2f} (azaltıldı)")
            elif key == ord('r'):
                # Min/max EAR sıfırla
                blink_detector.reset_ear_stats()
                print("EAR istatistikleri sıfırlandı")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()