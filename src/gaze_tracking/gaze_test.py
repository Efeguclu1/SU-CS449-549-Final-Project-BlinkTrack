"""
Gaze Tracking Test - İris tabanlı bakış takibi
Göz bebeği pozisyonunu kullanarak bakış yönünü tespit eder.
"""

import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

# İris landmark indeksleri (refine_landmarks=True gerektirir)
LEFT_IRIS = [468, 469, 470, 471, 472]   # Sol iris merkezi ve çevresi
RIGHT_IRIS = [473, 474, 475, 476, 477]  # Sağ iris merkezi ve çevresi

# Göz köşeleri (iris pozisyonunu referans almak için)
LEFT_EYE_INNER = 133   # Sol göz iç köşe
LEFT_EYE_OUTER = 33    # Sol göz dış köşe
RIGHT_EYE_INNER = 362  # Sağ göz iç köşe
RIGHT_EYE_OUTER = 263  # Sağ göz dış köşe

# Göz üst/alt kenarları
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374


def get_iris_center(landmarks, iris_indices, w, h):
    """İris merkezini hesapla"""
    points = []
    for idx in iris_indices:
        lm = landmarks[idx]
        points.append([lm.x * w, lm.y * h])
    return np.mean(points, axis=0)


def get_gaze_ratio(landmarks, w, h):
    """
    Bakış oranını hesapla (0-1 arası)
    0 = sola bakıyor, 0.5 = ortaya bakıyor, 1 = sağa bakıyor
    """
    # İris merkezlerini al
    left_iris = get_iris_center(landmarks, LEFT_IRIS, w, h)
    right_iris = get_iris_center(landmarks, RIGHT_IRIS, w, h)
    
    # Göz köşelerini al
    left_inner = np.array([landmarks[LEFT_EYE_INNER].x * w, landmarks[LEFT_EYE_INNER].y * h])
    left_outer = np.array([landmarks[LEFT_EYE_OUTER].x * w, landmarks[LEFT_EYE_OUTER].y * h])
    right_inner = np.array([landmarks[RIGHT_EYE_INNER].x * w, landmarks[RIGHT_EYE_INNER].y * h])
    right_outer = np.array([landmarks[RIGHT_EYE_OUTER].x * w, landmarks[RIGHT_EYE_OUTER].y * h])
    
    # Sol göz için yatay oran
    left_eye_width = np.linalg.norm(left_inner - left_outer)
    left_iris_pos = (left_iris[0] - left_outer[0]) / left_eye_width if left_eye_width > 0 else 0.5
    
    # Sağ göz için yatay oran
    right_eye_width = np.linalg.norm(right_inner - right_outer)
    right_iris_pos = (right_iris[0] - right_outer[0]) / right_eye_width if right_eye_width > 0 else 0.5
    
    # İki gözün ortalaması
    gaze_x = (left_iris_pos + right_iris_pos) / 2
    
    # Dikey bakış için
    left_top = np.array([landmarks[LEFT_EYE_TOP].x * w, landmarks[LEFT_EYE_TOP].y * h])
    left_bottom = np.array([landmarks[LEFT_EYE_BOTTOM].x * w, landmarks[LEFT_EYE_BOTTOM].y * h])
    right_top = np.array([landmarks[RIGHT_EYE_TOP].x * w, landmarks[RIGHT_EYE_TOP].y * h])
    right_bottom = np.array([landmarks[RIGHT_EYE_BOTTOM].x * w, landmarks[RIGHT_EYE_BOTTOM].y * h])
    
    left_eye_height = np.linalg.norm(left_top - left_bottom)
    right_eye_height = np.linalg.norm(right_top - right_bottom)
    
    left_center_y = (left_top[1] + left_bottom[1]) / 2
    right_center_y = (right_top[1] + right_bottom[1]) / 2
    
    left_gaze_y = (left_iris[1] - left_center_y) / (left_eye_height / 2) if left_eye_height > 0 else 0
    right_gaze_y = (right_iris[1] - right_center_y) / (right_eye_height / 2) if right_eye_height > 0 else 0
    
    gaze_y = (left_gaze_y + right_gaze_y) / 2
    
    return gaze_x, gaze_y


def main():
    cap = cv2.VideoCapture(0)
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,  # İris landmark'ları için gerekli!
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Ayna efekti (doğal his için)
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # İris merkezlerini çiz
                left_iris = get_iris_center(landmarks, LEFT_IRIS, w, h)
                right_iris = get_iris_center(landmarks, RIGHT_IRIS, w, h)
                
                cv2.circle(frame, (int(left_iris[0]), int(left_iris[1])), 3, (0, 255, 0), -1)
                cv2.circle(frame, (int(right_iris[0]), int(right_iris[1])), 3, (0, 255, 0), -1)
                
                # Göz köşelerini çiz
                for idx in [LEFT_EYE_INNER, LEFT_EYE_OUTER, RIGHT_EYE_INNER, RIGHT_EYE_OUTER]:
                    lm = landmarks[idx]
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (255, 0, 0), -1)
                
                # Bakış oranını hesapla
                gaze_x, gaze_y = get_gaze_ratio(landmarks, w, h)
                
                # Bakış yönünü göster
                direction = ""
                if gaze_x < 0.4:
                    direction = "SOLA"
                elif gaze_x > 0.6:
                    direction = "SAGA"
                else:
                    direction = "ORTA"
                
                if gaze_y < -0.3:
                    direction += " YUKARI"
                elif gaze_y > 0.3:
                    direction += " ASAGI"
                
                # Bilgileri ekrana yaz
                cv2.putText(frame, f"Gaze X: {gaze_x:.2f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Gaze Y: {gaze_y:.2f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Yon: {direction}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Bakış noktasını görselleştir (ekranın alt kısmında)
                viz_x = int(gaze_x * 200 + 50)
                viz_y = int(gaze_y * 50 + h - 80)
                cv2.rectangle(frame, (50, h - 130), (250, h - 30), (50, 50, 50), -1)
                cv2.circle(frame, (viz_x, viz_y), 8, (0, 0, 255), -1)
                cv2.putText(frame, "Bakis Noktasi", (80, h - 135),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.putText(frame, "Cikis: Q", (w - 100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow("Gaze Tracking Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()