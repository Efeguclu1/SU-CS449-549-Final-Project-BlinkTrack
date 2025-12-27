"""
BlinkTrack - Menu Navigation Demo
Bakış ile menü navigasyonu ve çift blink ile seçim
Figma prototype tasarımına uygun
"""

import cv2
import mediapipe as mp
import numpy as np
import time

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


class MenuItem:
    def __init__(self, text, icon=""):
        self.text = text
        self.icon = icon
        self.selected = False


class MenuDemo:
    def __init__(self, window_w=1000, window_h=700):
        self.window_w = window_w
        self.window_h = window_h
        self.center_x = window_w // 2
        self.center_y = window_h // 2
        
        # Menü öğeleri
        self.menu_items = [
            MenuItem("Option 1 - Ayarlar", "⚙"),
            MenuItem("Option 2 - Profil", "👤"),
            MenuItem("Option 3 - Yardim", "❓"),
            MenuItem("Option 4 - Cikis", "🚪"),
        ]
        
        # Menü ayarları
        self.menu_width = 350
        self.item_height = 70
        self.menu_padding = 15
        
        # Cursor
        self.cursor_x = self.center_x
        self.cursor_y = self.center_y
        
        # Durum
        self.hovered_index = None
        self.selected_index = None
        self.selection_time = None
        self.selection_log = []
        
        # Renkler (BGR)
        self.colors = {
            'bg': (35, 35, 35),
            'menu_bg': (50, 50, 50),
            'item_normal': (70, 70, 70),
            'item_hover': (80, 130, 80),
            'item_selected': (80, 180, 80),
            'text': (255, 255, 255),
            'cursor': (100, 100, 255),
            'title': (255, 200, 100),
        }
        
    def get_menu_bounds(self):
        """Menü sınırları"""
        total_height = len(self.menu_items) * self.item_height + 2 * self.menu_padding
        x = self.center_x - self.menu_width // 2
        y = self.center_y - total_height // 2
        return x, y, self.menu_width, total_height
    
    def get_item_bounds(self, index):
        """Menü öğesi sınırları"""
        menu_x, menu_y, menu_w, _ = self.get_menu_bounds()
        item_x = menu_x + self.menu_padding
        item_y = menu_y + self.menu_padding + index * self.item_height
        item_w = menu_w - 2 * self.menu_padding
        item_h = self.item_height - 8
        return item_x, item_y, item_w, item_h
    
    def get_hovered_item(self):
        """Cursor hangi öğenin üzerinde"""
        for i in range(len(self.menu_items)):
            x, y, w, h = self.get_item_bounds(i)
            if x <= self.cursor_x <= x + w and y <= self.cursor_y <= y + h:
                return i
        return None
    
    def select_item(self, index):
        """Öğe seç"""
        if index is not None and 0 <= index < len(self.menu_items):
            self.selected_index = index
            self.selection_time = time.time()
            self.selection_log.append({
                'time': time.strftime("%H:%M:%S"),
                'item': self.menu_items[index].text
            })
            return True
        return False
    
    def draw(self, frame, camera_frame, ear, calibrated):
        """Demo ekranını çiz"""
        display = np.zeros((self.window_h, self.window_w, 3), dtype=np.uint8)
        display[:] = self.colors['bg']
        
        # Başlık
        title = "BlinkTrack - Menu Navigation Demo"
        cv2.putText(display, title, (self.center_x - 200, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.colors['title'], 2)
        
        # Alt başlık
        subtitle = "Bak = Odaklan | Cift Blink = Sec"
        cv2.putText(display, subtitle, (self.center_x - 140, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Menü arka planı
        mx, my, mw, mh = self.get_menu_bounds()
        cv2.rectangle(display, (mx, my), (mx + mw, my + mh), self.colors['menu_bg'], -1)
        cv2.rectangle(display, (mx, my), (mx + mw, my + mh), (100, 100, 100), 2)
        
        # Seçim geri bildirimi temizle
        if self.selection_time and time.time() - self.selection_time > 1.0:
            self.selected_index = None
            self.selection_time = None
        
        # Hangi öğe üzerinde
        self.hovered_index = self.get_hovered_item()
        
        # Menü öğelerini çiz
        for i, item in enumerate(self.menu_items):
            ix, iy, iw, ih = self.get_item_bounds(i)
            
            # Renk belirle
            if self.selected_index == i:
                color = self.colors['item_selected']
            elif self.hovered_index == i:
                color = self.colors['item_hover']
            else:
                color = self.colors['item_normal']
            
            # Öğe arka planı
            cv2.rectangle(display, (ix, iy), (ix + iw, iy + ih), color, -1)
            cv2.rectangle(display, (ix, iy), (ix + iw, iy + ih), (120, 120, 120), 1)
            
            # Metin
            text_y = iy + ih // 2 + 8
            cv2.putText(display, item.text, (ix + 20, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.colors['text'], 2)
            
            # Seçim işareti
            if self.selected_index == i:
                cv2.putText(display, "OK", (ix + iw - 45, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Hover göstergesi
            if self.hovered_index == i and self.selected_index != i:
                cv2.circle(display, (ix + iw - 25, iy + ih // 2), 8, (0, 255, 255), 2)
        
        # Cursor (gaze point)
        cv2.circle(display, (self.cursor_x, self.cursor_y), 18, self.colors['cursor'], 2)
        cv2.circle(display, (self.cursor_x, self.cursor_y), 6, self.colors['cursor'], -1)
        
        # Durum göstergeleri
        self._draw_status(display, ear, calibrated)
        
        # Kamera önizleme
        cam_h, cam_w = 100, 133
        cam_resized = cv2.resize(camera_frame, (cam_w, cam_h))
        display[self.window_h - cam_h - 10:self.window_h - 10, 10:10 + cam_w] = cam_resized
        cv2.rectangle(display, (10, self.window_h - cam_h - 10), (10 + cam_w, self.window_h - 10), (100,100,100), 1)
        
        # Seçim logu
        self._draw_log(display)
        
        # Kontroller
        cv2.putText(display, "C: Kalibrasyon | T/G: Esik | Q: Cikis", 
                   (self.window_w - 300, self.window_h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        
        return display
    
    def _draw_status(self, display, ear, calibrated):
        """Durum göstergeleri"""
        # Sağ üst köşe
        x = self.window_w - 180
        y = 30
        
        # Kalibrasyon
        cal_color = (0, 255, 0) if calibrated else (0, 0, 255)
        cal_text = "Kalibre: OK" if calibrated else "Kalibre: GEREKLI"
        cv2.putText(display, cal_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cal_color, 1)
        
        # EAR
        ear_color = (0, 255, 0) if ear >= 0.24 else (0, 0, 255)
        cv2.putText(display, f"EAR: {ear:.2f}", (x, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ear_color, 1)
        
        # Hover durumu
        if self.hovered_index is not None:
            cv2.putText(display, f"Hover: {self.hovered_index + 1}", (x, y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def _draw_log(self, display):
        """Seçim logu"""
        if not self.selection_log:
            return
        
        x = self.window_w - 200
        y = 120
        
        cv2.putText(display, "Secim Logu:", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Son 5 seçim
        for i, log in enumerate(self.selection_log[-5:]):
            cv2.putText(display, f"{log['time']} - {log['item'][:15]}", (x, y + 20 + i * 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 100), 1)


def main():
    cap = cv2.VideoCapture(0)
    
    demo = MenuDemo(window_w=1000, window_h=700)
    calibration = Calibration()
    blink_detector = BlinkDetector(threshold=0.24)
    
    smoothing = 0.25      # Daha hızlı tepki
    sensitivity = 1.5     # Daha hassas
    prev_x, prev_y = demo.center_x, demo.center_y
    
    calib_mode = False
    calib_samples = []
    
    window_name = "BlinkTrack - Menu Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, demo.window_w, demo.window_h)
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        print("\n" + "="*50)
        print("   BlinkTrack - Menu Navigation Demo")
        print("="*50)
        print("\nC: Kalibrasyon | Q: Çıkış")
        print("Çift blink ile seçim yapın!")
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
                
                # Kalibrasyon
                if calib_mode:
                    calib_samples.append((gaze_x, gaze_y))
                    if len(calib_samples) >= 30:
                        avg_x = np.mean([s[0] for s in calib_samples])
                        avg_y = np.mean([s[1] for s in calib_samples])
                        calibration.calibrate(avg_x, avg_y)
                        calib_mode = False
                        calib_samples = []
                        print("✓ Kalibrasyon tamamlandı!")
                
                # Cursor güncelle
                if calibration.calibrated:
                    target_x, target_y = calibration.map_to_screen(
                        gaze_x, gaze_y, demo.window_w, demo.window_h, sensitivity
                    )
                    demo.cursor_x = int(prev_x + (target_x - prev_x) * smoothing)
                    demo.cursor_y = int(prev_y + (target_y - prev_y) * smoothing)
                    prev_x, prev_y = demo.cursor_x, demo.cursor_y
                
                # Blink
                blink = blink_detector.update(ear)
                
                if blink == "double" and demo.hovered_index is not None:
                    demo.select_item(demo.hovered_index)
                    print(f"  ✓ Seçildi: {demo.menu_items[demo.hovered_index].text}")
            
            # Çiz
            display = demo.draw(frame, frame, ear, calibration.calibrated)
            
            # Kalibrasyon modu göstergesi
            if calib_mode:
                cv2.rectangle(display, (0, 0), (demo.window_w, demo.window_h), (0, 100, 100), 5)
                cv2.putText(display, f"Kalibre ediliyor... {len(calib_samples)}/30", 
                           (demo.center_x - 120, demo.center_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.circle(display, (demo.center_x, demo.center_y - 50), 25, (0, 255, 255), 3)
            
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                calib_mode = True
                calib_samples = []
                print("Kalibrasyon başladı - Ekranın ortasına bakın...")
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