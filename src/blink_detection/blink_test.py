"""
Blink Detection Test - Tek ve çift göz kırpma algılama
Doğru EAR eşik değerleri ile çalışır.
"""

import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh

# Göz landmark indeksleri (EAR hesaplama için)
# Her göz için 6 nokta: [dış köşe, üst1, üst2, iç köşe, alt1, alt2]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]


def euclidean_dist(a, b):
    """İki nokta arası mesafe"""
    return np.linalg.norm(np.array(a) - np.array(b))


def eye_aspect_ratio(eye_points):
    """
    Eye Aspect Ratio (EAR) hesapla
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    
    Açık göz: ~0.25-0.35
    Kapalı göz: ~0.15-0.20
    """
    # Dikey mesafeler
    vertical_1 = euclidean_dist(eye_points[1], eye_points[5])
    vertical_2 = euclidean_dist(eye_points[2], eye_points[4])
    
    # Yatay mesafe
    horizontal = euclidean_dist(eye_points[0], eye_points[3])
    
    if horizontal == 0:
        return 0.0
    
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


class BlinkDetector:
    """Tek, çift ve uzun blink algılama"""
    
    def __init__(self, ear_threshold=0.21):
        self.ear_threshold = ear_threshold
        
        # Durum takibi
        self.eye_closed = False
        self.close_start_time = None
        self.last_blink_time = None
        self.pending_single = False
        
        # Zamanlama ayarları (saniye)
        self.max_single_duration = 0.35      # Tek blink max süresi
        self.double_blink_window = 0.45      # Çift blink için zaman penceresi
        self.prolonged_duration = 0.7        # Uzun blink süresi
        
    def update(self, ear):
        """
        EAR değeri ile güncelle
        Return: "single", "double", "prolonged", veya None
        """
        current_time = time.time()
        detected = None
        
        eye_now_closed = ear < self.ear_threshold
        
        # Göz yeni kapandı
        if eye_now_closed and not self.eye_closed:
            self.eye_closed = True
            self.close_start_time = current_time
        
        # Göz yeni açıldı
        elif not eye_now_closed and self.eye_closed:
            self.eye_closed = False
            
            if self.close_start_time is not None:
                duration = current_time - self.close_start_time
                
                # Uzun blink
                if duration >= self.prolonged_duration:
                    detected = "prolonged"
                    self.pending_single = False
                
                # Normal blink (tek veya çiftin parçası)
                elif duration <= self.max_single_duration:
                    # Çift blink kontrolü
                    if (self.pending_single and 
                        self.last_blink_time is not None and
                        (current_time - self.last_blink_time) <= self.double_blink_window):
                        detected = "double"
                        self.pending_single = False
                    else:
                        # İlk blink - çift için bekle
                        self.pending_single = True
                        self.last_blink_time = current_time
            
            self.close_start_time = None
        
        # Çift blink penceresi geçti mi kontrol et
        if (self.pending_single and 
            self.last_blink_time is not None and
            (current_time - self.last_blink_time) > self.double_blink_window):
            detected = "single"
            self.pending_single = False
        
        return detected


def main():
    cap = cv2.VideoCapture(0)
    
    # Blink dedektörü
    blink_detector = BlinkDetector(ear_threshold=0.21)
    
    # Görsel geri bildirim için
    last_blink_type = None
    last_blink_display_time = 0
    blink_display_duration = 0.8  # Blink mesajını gösterme süresi
    
    # İstatistikler
    single_count = 0
    double_count = 0
    prolonged_count = 0
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        print("\n=== BlinkTrack Blink Detection ===")
        print("Göz kırpma türleri:")
        print("  - Tek blink: Normal kırpma")
        print("  - Çift blink: Hızlı iki kırpma (CLICK)")
        print("  - Uzun blink: Gözü 0.7sn+ kapalı tutma")
        print("\nÇıkış için Q basın\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            
            ear = 0.0
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Göz noktalarını al
                left_eye_pts = []
                right_eye_pts = []
                
                for idx in LEFT_EYE:
                    lm = landmarks[idx]
                    pt = (lm.x * w, lm.y * h)
                    left_eye_pts.append(pt)
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
                
                for idx in RIGHT_EYE:
                    lm = landmarks[idx]
                    pt = (lm.x * w, lm.y * h)
                    right_eye_pts.append(pt)
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
                
                # EAR hesapla
                left_ear = eye_aspect_ratio(left_eye_pts)
                right_ear = eye_aspect_ratio(right_eye_pts)
                ear = (left_ear + right_ear) / 2.0
                
                # Blink algılama
                blink = blink_detector.update(ear)
                
                if blink:
                    last_blink_type = blink
                    last_blink_display_time = time.time()
                    
                    if blink == "single":
                        single_count += 1
                        print(f"[TEK BLINK] Toplam: {single_count}")
                    elif blink == "double":
                        double_count += 1
                        print(f"[ÇİFT BLINK - CLICK!] Toplam: {double_count}")
                    elif blink == "prolonged":
                        prolonged_count += 1
                        print(f"[UZUN BLINK] Toplam: {prolonged_count}")
            
            # EAR göster
            ear_color = (0, 255, 0) if ear >= blink_detector.ear_threshold else (0, 0, 255)
            cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, ear_color, 2)
            
            # Eşik değeri
            cv2.putText(frame, f"Esik: {blink_detector.ear_threshold}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Göz durumu
            eye_status = "KAPALI" if ear < blink_detector.ear_threshold else "ACIK"
            cv2.putText(frame, f"Goz: {eye_status}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, ear_color, 2)
            
            # Blink geri bildirimi
            if last_blink_type and (time.time() - last_blink_display_time) < blink_display_duration:
                if last_blink_type == "single":
                    text = "TEK BLINK"
                    color = (0, 255, 255)
                elif last_blink_type == "double":
                    text = "CIFT BLINK - CLICK!"
                    color = (0, 255, 0)
                else:
                    text = "UZUN BLINK"
                    color = (255, 0, 255)
                
                # Büyük bildirim
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                text_x = (w - text_size[0]) // 2
                cv2.putText(frame, text, (text_x, h // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            # İstatistikler
            cv2.putText(frame, f"Tek: {single_count} | Cift: {double_count} | Uzun: {prolonged_count}",
                       (10, h - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            # Talimatlar
            cv2.putText(frame, "Cift blink = Click | Q: Cikis", (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            # EAR çubuğu (görsel gösterge)
            bar_width = 200
            bar_height = 20
            bar_x = w - bar_width - 20
            bar_y = 20
            
            # Arka plan
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                         (50, 50, 50), -1)
            
            # EAR seviyesi
            ear_width = int(min(ear / 0.4, 1.0) * bar_width)  # 0.4 max olarak
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + ear_width, bar_y + bar_height),
                         ear_color, -1)
            
            # Eşik çizgisi
            threshold_x = bar_x + int(blink_detector.ear_threshold / 0.4 * bar_width)
            cv2.line(frame, (threshold_x, bar_y), (threshold_x, bar_y + bar_height),
                    (255, 255, 255), 2)
            
            cv2.imshow("Blink Detection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                blink_detector.ear_threshold = min(0.35, blink_detector.ear_threshold + 0.01)
                print(f"Eşik artırıldı: {blink_detector.ear_threshold:.2f}")
            elif key == ord('-'):
                blink_detector.ear_threshold = max(0.15, blink_detector.ear_threshold - 0.01)
                print(f"Eşik azaltıldı: {blink_detector.ear_threshold:.2f}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n=== Sonuçlar ===")
    print(f"Tek blink: {single_count}")
    print(f"Çift blink: {double_count}")
    print(f"Uzun blink: {prolonged_count}")


if __name__ == "__main__":
    main()