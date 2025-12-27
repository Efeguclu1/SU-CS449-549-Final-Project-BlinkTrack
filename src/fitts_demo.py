"""
BlinkTrack - Fitts' Law Target Demo
Hedef seçme görevi ile performans ölçümü
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import csv
import random
from datetime import datetime

mp_face_mesh = mp.solutions.face_mesh

# === LANDMARK İNDEKSLERİ ===
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_INNER, LEFT_EYE_OUTER = 133, 33
RIGHT_EYE_INNER, RIGHT_EYE_OUTER = 362, 263
LEFT_EYE_TOP, LEFT_EYE_BOTTOM = 159, 145
RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM = 386, 374
LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR = [263, 387, 385, 362, 380, 373]


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
    left_iris = get_iris_center(landmarks, LEFT_IRIS, w, h)
    right_iris = get_iris_center(landmarks, RIGHT_IRIS, w, h)
    
    l_inner = np.array([landmarks[LEFT_EYE_INNER].x * w, landmarks[LEFT_EYE_INNER].y * h])
    l_outer = np.array([landmarks[LEFT_EYE_OUTER].x * w, landmarks[LEFT_EYE_OUTER].y * h])
    r_inner = np.array([landmarks[RIGHT_EYE_INNER].x * w, landmarks[RIGHT_EYE_INNER].y * h])
    r_outer = np.array([landmarks[RIGHT_EYE_OUTER].x * w, landmarks[RIGHT_EYE_OUTER].y * h])
    
    l_width = np.linalg.norm(l_inner - l_outer)
    r_width = np.linalg.norm(r_inner - r_outer)
    
    l_ratio = (left_iris[0] - l_outer[0]) / l_width if l_width > 0 else 0.5
    r_ratio = (right_iris[0] - r_outer[0]) / r_width if r_width > 0 else 0.5
    gaze_x = (l_ratio + r_ratio) / 2
    
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


class Calibration:
    def __init__(self):
        self.calibrated = False
        self.center_x = 0.5
        self.center_y = 0.0
        
    def calibrate(self, gaze_x, gaze_y):
        self.center_x = gaze_x
        self.center_y = gaze_y
        self.calibrated = True
        print(f"  Kalibrasyon merkezi: ({gaze_x:.3f}, {gaze_y:.3f})")
        
    def map_to_screen(self, gaze_x, gaze_y, screen_w, screen_h, sensitivity=1.0):
        if not self.calibrated:
            return screen_w // 2, screen_h // 2
        
        # Merkeze göre fark hesapla
        diff_x = gaze_x - self.center_x
        diff_y = gaze_y - self.center_y
        
        # Ekran koordinatına çevir
        # Sola bakınca (diff_x azalır) cursor sola gitmeli
        # Yukarı bakınca (diff_y azalır) cursor yukarı gitmeli
        
        # Hassasiyet ve ölçekleme
        scale_x = 8.0 * sensitivity  # Yatay hareket çarpanı
        scale_y = 6.0 * sensitivity  # Dikey hareket çarpanı
        
        # Merkez + fark (X ters çünkü webcam ayna)
        norm_x = 0.5 + diff_x * scale_x
        norm_y = 0.5 + diff_y * scale_y
        
        # Sınırla
        norm_x = max(0.02, min(0.98, norm_x))
        norm_y = max(0.02, min(0.98, norm_y))
        
        return int(norm_x * screen_w), int(norm_y * screen_h)


class BlinkDetector:
    def __init__(self, threshold=0.24):
        self.threshold = threshold
        self.eye_closed = False
        self.close_time = None
        self.last_blink = None
        self.pending = False
        self.max_single = 0.4
        self.double_window = 0.55
        self.prolonged = 0.8
        
    def update(self, ear):
        now = time.time()
        result = None
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
        
        if self.pending and self.last_blink and (now - self.last_blink) > self.double_window:
            result = "single"
            self.pending = False
        
        return result


class FittsLawDemo:
    def __init__(self, window_w=1200, window_h=800):
        self.window_w = window_w
        self.window_h = window_h
        self.center_x = window_w // 2
        self.center_y = window_h // 2
        
        # Hedef ayarları (Fitts' Law parametreleri)
        self.target_sizes = [50, 70, 90]        # Piksel (W - width)
        self.target_distances = [150, 250, 350]  # Piksel (D - distance)
        self.trials_per_condition = 2            # Her koşul için deneme
        
        # Deneme verileri
        self.trials = []
        self.results = []
        self.current_trial = 0
        
        # Durum
        self.state = "waiting"  # waiting, calibrating, countdown, running, success, finished
        self.target_pos = None
        self.target_size = 60
        self.trial_start_time = None
        
        # Cursor
        self.cursor_x = self.center_x
        self.cursor_y = self.center_y
        
        # Geri bildirim
        self.feedback_time = None
        self.feedback_type = None
        
    def generate_trials(self):
        """Rastgele deneme sırası oluştur"""
        self.trials = []
        for size in self.target_sizes:
            for distance in self.target_distances:
                for _ in range(self.trials_per_condition):
                    self.trials.append({'size': size, 'distance': distance})
        random.shuffle(self.trials)
        
    def get_target_position(self, distance):
        """Rastgele açıda hedef pozisyonu"""
        angle = random.uniform(0, 2 * np.pi)
        x = int(self.center_x + distance * np.cos(angle))
        y = int(self.center_y + distance * np.sin(angle))
        
        # Ekran sınırları
        margin = 80
        x = max(margin, min(self.window_w - margin, x))
        y = max(margin, min(self.window_h - margin, y))
        
        return x, y
    
    def start_trial(self):
        """Yeni deneme başlat"""
        if self.current_trial >= len(self.trials):
            self.state = "finished"
            return
        
        trial = self.trials[self.current_trial]
        self.target_size = trial['size']
        self.target_pos = self.get_target_position(trial['distance'])
        self.trial_start_time = time.time()
        self.state = "running"
        
    def check_hit(self):
        """Cursor hedefte mi?"""
        if self.target_pos is None:
            return False, 0
        
        dx = self.cursor_x - self.target_pos[0]
        dy = self.cursor_y - self.target_pos[1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        return distance <= self.target_size / 2, distance
    
    def record_result(self, success, distance):
        """Sonucu kaydet"""
        trial = self.trials[self.current_trial]
        completion_time = (time.time() - self.trial_start_time) * 1000  # ms
        
        # Fitts' Law: ID = log2(D/W + 1)
        index_of_difficulty = np.log2(trial['distance'] / trial['size'] + 1)
        
        result = {
            'trial': self.current_trial + 1,
            'target_size': trial['size'],
            'target_distance': trial['distance'],
            'index_of_difficulty': round(index_of_difficulty, 3),
            'completion_time_ms': round(completion_time, 1),
            'success': success,
            'error_distance': round(distance, 1),
            'target_x': self.target_pos[0],
            'target_y': self.target_pos[1],
            'click_x': self.cursor_x,
            'click_y': self.cursor_y
        }
        self.results.append(result)
        
    def save_results(self):
        """Sonuçları CSV'ye kaydet"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fitts_results_{timestamp}.csv"
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            if self.results:
                writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
                writer.writeheader()
                writer.writerows(self.results)
        
        return filename
    
    def get_statistics(self):
        """İstatistikleri hesapla"""
        if not self.results:
            return {}
        
        successful = [r for r in self.results if r['success']]
        
        stats = {
            'total_trials': len(self.results),
            'successful': len(successful),
            'success_rate': len(successful) / len(self.results) * 100,
            'avg_time_ms': np.mean([r['completion_time_ms'] for r in successful]) if successful else 0,
            'std_time_ms': np.std([r['completion_time_ms'] for r in successful]) if successful else 0,
        }
        
        return stats
    
    def draw(self, frame, camera_frame, ear):
        """Demo ekranını çiz"""
        # Arka plan
        display = np.zeros((self.window_h, self.window_w, 3), dtype=np.uint8)
        display[:] = (40, 40, 40)
        
        if self.state == "waiting":
            self._draw_waiting(display)
        elif self.state == "calibrating":
            self._draw_calibrating(display)
        elif self.state == "countdown":
            self._draw_countdown(display)
        elif self.state == "running":
            self._draw_running(display)
        elif self.state == "success":
            self._draw_success(display)
        elif self.state == "finished":
            self._draw_finished(display)
        
        # Kamera önizleme
        cam_h, cam_w = 120, 160
        cam_resized = cv2.resize(camera_frame, (cam_w, cam_h))
        display[10:10+cam_h, self.window_w-cam_w-10:self.window_w-10] = cam_resized
        
        # EAR göstergesi
        ear_color = (0, 255, 0) if ear >= 0.24 else (0, 0, 255)
        cv2.putText(display, f"EAR: {ear:.2f}", (self.window_w - cam_w - 5, cam_h + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, ear_color, 1)
        
        return display
    
    def _draw_waiting(self, display):
        """Başlangıç ekranı"""
        texts = [
            "FITTS' LAW HEDEF SECME DEMO",
            "",
            "Bu demo bakis tabanli hedef secme performansini olcer.",
            "",
            f"Toplam deneme: {len(self.trials) if self.trials else '?'}",
            f"Hedef boyutlari: {self.target_sizes}",
            f"Hedef mesafeleri: {self.target_distances}",
            "",
            "Talimatlar:",
            "1. Yesil hedefe bakin",
            "2. Cursor hedefe gelince CIFT BLINK yapin",
            "3. Hizli ve dogru olmaya calisin!",
            "",
            "[SPACE] - Kalibrasyona Basla",
            "[Q] - Cikis"
        ]
        
        y = 120
        for text in texts:
            size = 0.7 if text and text[0] not in ['[', ''] else 0.6
            color = (0, 255, 255) if text.startswith('[') else (255, 255, 255)
            cv2.putText(display, text, (50, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 2)
            y += 35
    
    def _draw_calibrating(self, display):
        """Kalibrasyon ekranı"""
        cv2.circle(display, (self.center_x, self.center_y), 30, (0, 255, 255), 3)
        cv2.circle(display, (self.center_x, self.center_y), 8, (0, 255, 255), -1)
        
        cv2.putText(display, "KALIBRASYON", (self.center_x - 80, self.center_y - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(display, "Ortadaki noktaya bakin...", (self.center_x - 120, self.center_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def _draw_countdown(self, display):
        """Geri sayım"""
        remaining = 3 - int(time.time() - self.countdown_start)
        if remaining > 0:
            cv2.putText(display, str(remaining), (self.center_x - 30, self.center_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
        cv2.putText(display, "Hazir olun!", (self.center_x - 70, self.center_y + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    def _draw_running(self, display):
        """Aktif deneme"""
        if self.target_pos:
            tx, ty = self.target_pos
            radius = self.target_size // 2
            
            # Cursor hedefte mi?
            dx = self.cursor_x - tx
            dy = self.cursor_y - ty
            dist = np.sqrt(dx*dx + dy*dy)
            on_target = dist <= radius
            
            # Hedef
            color = (0, 255, 200) if on_target else (0, 200, 100)
            cv2.circle(display, (tx, ty), radius, color, -1)
            cv2.circle(display, (tx, ty), radius, (255, 255, 255), 2)
            cv2.circle(display, (tx, ty), 5, (255, 255, 255), -1)
            
            # Cursor
            cv2.circle(display, (self.cursor_x, self.cursor_y), 12, (100, 100, 255), -1)
            cv2.circle(display, (self.cursor_x, self.cursor_y), 12, (255, 255, 255), 2)
            cv2.line(display, (self.cursor_x - 8, self.cursor_y), (self.cursor_x + 8, self.cursor_y), (255,255,255), 1)
            cv2.line(display, (self.cursor_x, self.cursor_y - 8), (self.cursor_x, self.cursor_y + 8), (255,255,255), 1)
        
        # Üst bilgi
        cv2.putText(display, f"Deneme: {self.current_trial + 1}/{len(self.trials)}", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        elapsed = int((time.time() - self.trial_start_time) * 1000)
        cv2.putText(display, f"Sure: {elapsed} ms", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Talimat
        cv2.putText(display, "Hedefe bakin ve CIFT BLINK yapin!", (self.center_x - 180, self.window_h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    def _draw_success(self, display):
        """Başarı geri bildirimi"""
        if self.target_pos:
            tx, ty = self.target_pos
            cv2.circle(display, (tx, ty), self.target_size // 2 + 10, (0, 255, 255), 4)
        
        cv2.putText(display, "BASARILI!", (self.center_x - 80, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        if self.results:
            last = self.results[-1]
            cv2.putText(display, f"{last['completion_time_ms']:.0f} ms", (self.center_x - 50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    def _draw_finished(self, display):
        """Bitiş ekranı"""
        stats = self.get_statistics()
        
        texts = [
            "DEMO TAMAMLANDI!",
            "",
            f"Toplam Deneme: {stats.get('total_trials', 0)}",
            f"Basarili: {stats.get('successful', 0)}",
            f"Basari Orani: {stats.get('success_rate', 0):.1f}%",
            f"Ortalama Sure: {stats.get('avg_time_ms', 0):.0f} ms",
            f"Std Sapma: {stats.get('std_time_ms', 0):.0f} ms",
            "",
            "[S] - Sonuclari Kaydet (CSV)",
            "[R] - Tekrar Basla",
            "[Q] - Cikis"
        ]
        
        y = 150
        for text in texts:
            color = (0, 255, 0) if "TAMAMLANDI" in text else (0, 255, 255) if text.startswith('[') else (255, 255, 255)
            cv2.putText(display, text, (self.center_x - 150, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y += 40


def main():
    cap = cv2.VideoCapture(0)
    
    demo = FittsLawDemo(window_w=1200, window_h=800)
    demo.generate_trials()
    
    calibration = Calibration()
    blink_detector = BlinkDetector(threshold=0.24)
    
    # Ayarlar
    smoothing = 0.3        # Hızlı ama yumuşak tepki
    sensitivity = 1.0      # Başlangıç değeri (scale zaten 8x)
    prev_x, prev_y = demo.center_x, demo.center_y
    
    # Kalibrasyon
    calib_samples = []
    
    window_name = "BlinkTrack - Fitts' Law Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, demo.window_w, demo.window_h)
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        print("\n" + "="*50)
        print("   BlinkTrack - Fitts' Law Demo")
        print("="*50)
        print("\nSPACE: Başla | Q: Çıkış")
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
                
                gaze_x, gaze_y = get_gaze_ratio(landmarks, w, h)
                
                left_pts = [[landmarks[i].x * w, landmarks[i].y * h] for i in LEFT_EYE_EAR]
                right_pts = [[landmarks[i].x * w, landmarks[i].y * h] for i in RIGHT_EYE_EAR]
                ear = (eye_aspect_ratio(left_pts) + eye_aspect_ratio(right_pts)) / 2
                
                # Kalibrasyon modu
                if demo.state == "calibrating":
                    calib_samples.append((gaze_x, gaze_y))
                    if len(calib_samples) >= 30:
                        avg_x = np.mean([s[0] for s in calib_samples])
                        avg_y = np.mean([s[1] for s in calib_samples])
                        calibration.calibrate(avg_x, avg_y)
                        calib_samples = []
                        demo.state = "countdown"
                        demo.countdown_start = time.time()
                        print("Kalibrasyon tamamlandı!")
                
                # Cursor güncelle
                if calibration.calibrated:
                    target_x, target_y = calibration.map_to_screen(
                        gaze_x, gaze_y, demo.window_w, demo.window_h, sensitivity
                    )
                    demo.cursor_x = int(prev_x + (target_x - prev_x) * smoothing)
                    demo.cursor_y = int(prev_y + (target_y - prev_y) * smoothing)
                    prev_x, prev_y = demo.cursor_x, demo.cursor_y
                
                # Blink algılama
                blink = blink_detector.update(ear)
                
                if blink == "double" and demo.state == "running":
                    success, dist = demo.check_hit()
                    demo.record_result(success, dist)
                    
                    if success:
                        demo.state = "success"
                        demo.feedback_time = time.time()
                        print(f"  ✓ Deneme {demo.current_trial + 1}: {demo.results[-1]['completion_time_ms']:.0f} ms")
                    else:
                        demo.current_trial += 1
                        demo.start_trial()
                        print(f"  ✗ Deneme {demo.current_trial}: Iskalandı")
            
            # Countdown kontrolü
            if demo.state == "countdown":
                if time.time() - demo.countdown_start >= 3:
                    demo.start_trial()
            
            # Success sonrası geçiş
            if demo.state == "success" and demo.feedback_time:
                if time.time() - demo.feedback_time >= 0.6:
                    demo.current_trial += 1
                    demo.start_trial()
            
            # Çiz
            display = demo.draw(frame, frame, ear)
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                if demo.state == "waiting":
                    demo.state = "calibrating"
                    calib_samples = []
                    print("Kalibrasyon başladı...")
            elif key == ord('s') and demo.state == "finished":
                filename = demo.save_results()
                print(f"Sonuçlar kaydedildi: {filename}")
            elif key == ord('r') and demo.state == "finished":
                demo.current_trial = 0
                demo.results = []
                demo.generate_trials()
                demo.state = "waiting"
                print("Demo sıfırlandı")
            elif key == ord('t'):
                blink_detector.threshold = min(0.35, blink_detector.threshold + 0.01)
                print(f"EAR Eşik: {blink_detector.threshold:.2f}")
            elif key == ord('g'):
                blink_detector.threshold = max(0.15, blink_detector.threshold - 0.01)
                print(f"EAR Eşik: {blink_detector.threshold:.2f}")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()