import cv2
import mediapipe as mp
import numpy as np
import math
from enum import Enum
import time
import pygame
import threading

class Activity(Enum):
    SITTING = "Duduk"
    STANDING = "Berdiri"
    WALKING = "Berjalan"
    LYING = "Tidur"
    SQUATTING = "Jongkok"
    FALLING = "JATUH!"
    UNKNOWN = "Tidak Dikenal"

class AlarmManager:
    def __init__(self):
        pygame.mixer.init()
        self.is_playing = False
        self.alarm_thread = None
        self.create_alarm_sound()
        
    def create_alarm_sound(self):
        """Membuat suara alarm sederhana"""
        try:
            # Coba load file suara eksternal jika ada
            self.alarm_sound = pygame.mixer.Sound("alarm.wav")
            print("File alarm.wav loaded successfully")
        except:
            # Jika tidak ada file, buat suara alarm sederhana
            print("Creating default alarm sound...")
            try:
                sample_rate = 44100
                duration = 0.3  # Lebih pendek
                frequency = 880
                frames = int(duration * sample_rate)
                
                # Buat array suara yang lebih sederhana
                t = np.linspace(0, duration, frames, False)
                wave = 0.5 * np.sin(2 * np.pi * frequency * t)
                
                # Convert ke 16-bit integers
                wave = (wave * 32767).astype(np.int16)
                
                # Buat stereo
                stereo_wave = np.column_stack((wave, wave))
                stereo_wave = np.ascontiguousarray(stereo_wave)
                
                self.alarm_sound = pygame.sndarray.make_sound(stereo_wave)
                print("Default alarm sound created successfully")
            except Exception as e:
                print(f"Error creating alarm sound: {e}")
                # Fallback ke beep system
                self.alarm_sound = None
    
    def play_alarm(self):
        """Memainkan alarm dalam thread terpisah"""
        if not self.is_playing and self.alarm_sound is not None:
            self.is_playing = True
            self.alarm_thread = threading.Thread(target=self._alarm_loop)
            self.alarm_thread.daemon = True
            self.alarm_thread.start()
        elif self.alarm_sound is None:
            # Fallback: print warning saja
            print("🚨 ALARM: JATUH TERDETEKSI! 🚨")
    
    def _alarm_loop(self):
        """Loop alarm yang berjalan di thread terpisah"""
        try:
            while self.is_playing:
                self.alarm_sound.play()
                pygame.time.wait(600)  # Beep lebih lama
                pygame.time.wait(300)  # Jeda lebih pendek
        except Exception as e:
            print(f"Error in alarm loop: {e}")
    
    def stop_alarm(self):
        """Menghentikan alarm"""
        self.is_playing = False
        if self.alarm_sound is not None:
            self.alarm_sound.stop()

class RehabActivityDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Inisialisasi alarm manager
        self.alarm_manager = AlarmManager()
        
        # Variabel untuk deteksi jatuh
        self.fall_detected = False
        self.fall_start_time = None
        self.last_activity = Activity.UNKNOWN
        self.velocity_threshold = 0.3
        self.fall_duration_threshold = 2.0
        
    def calculate_angle(self, point1, point2, point3):
        """Menghitung sudut antara tiga titik"""
        vector1 = [point1.x - point2.x, point1.y - point2.y]
        vector2 = [point3.x - point2.x, point3.y - point2.y]
        
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
        magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)
        
        if magnitude1 * magnitude2 == 0:
            return 0
            
        cosine_angle = dot_product / (magnitude1 * magnitude2)
        cosine_angle = max(-1, min(1, cosine_angle))
        angle = math.degrees(math.acos(cosine_angle))
        
        return angle
    
    def calculate_distance(self, point1, point2):
        """Menghitung jarak Euclidean antara dua titik"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def calculate_velocity(self, prev_landmarks, curr_landmarks, time_delta):
        """Menghitung kecepatan perubahan posisi landmarks"""
        if not prev_landmarks or not curr_landmarks:
            return 0
            
        total_distance = 0
        valid_points = 0
        
        for i in range(len(curr_landmarks)):
            if i < len(prev_landmarks):
                distance = self.calculate_distance(prev_landmarks[i], curr_landmarks[i])
                total_distance += distance
                valid_points += 1
        
        if valid_points == 0 or time_delta == 0:
            return 0
            
        avg_velocity = total_distance / valid_points / time_delta
        return avg_velocity
    
    def detect_fall(self, landmarks, prev_landmarks, time_delta):
        """Mendeteksi apakah terjadi jatuh"""
        if not landmarks or not prev_landmarks:
            return False, 0
        
        # Hitung kecepatan perubahan posisi
        velocity = self.calculate_velocity(prev_landmarks, landmarks, time_delta)
        
        # Ambil landmark penting untuk deteksi jatuh
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        # Hitung tinggi tubuh (dari bahu ke pergelangan kaki)
        body_height_left = abs(left_shoulder.y - left_ankle.y)
        body_height_right = abs(right_shoulder.y - right_ankle.y)
        avg_body_height = (body_height_left + body_height_right) / 2
        
        # Hitung orientasi tubuh
        vertical_orientation = abs(left_shoulder.y - left_hip.y)
        horizontal_orientation = abs(left_shoulder.x - left_hip.x)
        
        # Kriteria deteksi jatuh:
        head_y = min(left_shoulder.y, right_shoulder.y)
        hip_y = max(left_hip.y, right_hip.y)
        
        sudden_movement = velocity > self.velocity_threshold
        significant_height_drop = avg_body_height < 0.4
        horizontal_body = vertical_orientation < 0.15
        head_below_hips = head_y > hip_y
        
        # Jatuh terdeteksi jika memenuhi kriteria tertentu
        fall_detected = (sudden_movement and (significant_height_drop or horizontal_body)) or \
                       (head_below_hips and horizontal_body)
        
        return fall_detected, velocity
    
    def detect_activity(self, landmarks, prev_landmarks=None, time_delta=0.033):
        """Mendeteksi aktivitas berdasarkan pose landmarks"""
        if not landmarks:
            return Activity.UNKNOWN
        
        # Deteksi jatuh terlebih dahulu (prioritas tinggi)
        fall_detected = False
        if prev_landmarks:
            fall_detected, velocity = self.detect_fall(landmarks, prev_landmarks, time_delta)
            if fall_detected:
                if not self.fall_detected:
                    self.fall_detected = True
                    self.fall_start_time = time.time()
                    # Trigger alarm ketika jatuh terdeteksi
                    self.alarm_manager.play_alarm()
                    return Activity.FALLING
                else:
                    # Jika sudah terdeteksi jatuh, pertahankan status jatuh untuk beberapa waktu
                    if time.time() - self.fall_start_time < self.fall_duration_threshold:
                        return Activity.FALLING
                    else:
                        self.fall_detected = False
                        self.fall_start_time = None
                        self.alarm_manager.stop_alarm()
        
        # Reset status jatuh jika tidak terdeteksi lagi
        if self.fall_detected and not fall_detected:
            self.fall_detected = False
            self.alarm_manager.stop_alarm()
        
        # Ambil landmark penting
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        # Hitung sudut-sudut penting
        hip_angle_left = self.calculate_angle(left_shoulder, left_hip, left_knee)
        hip_angle_right = self.calculate_angle(right_shoulder, right_hip, right_knee)
        knee_angle_left = self.calculate_angle(left_hip, left_knee, left_ankle)
        knee_angle_right = self.calculate_angle(right_hip, right_knee, right_ankle)
        
        # Hitung perbedaan tinggi antara bahu dan pinggul
        shoulder_hip_diff = abs((left_shoulder.y + right_shoulder.y)/2 - (left_hip.y + right_hip.y)/2)
        
        # Hitung posisi relatif kaki
        ankle_height_diff = abs(left_ankle.y - right_ankle.y)
        
        # Hitung tinggi tubuh (dari bahu ke pergelangan kaki)
        body_height_left = abs(left_shoulder.y - left_ankle.y)
        body_height_right = abs(right_shoulder.y - right_ankle.y)
        avg_body_height = (body_height_left + body_height_right) / 2
        
        # Deteksi aktivitas berdasarkan aturan
        avg_hip_angle = (hip_angle_left + hip_angle_right) / 2
        avg_knee_angle = (knee_angle_left + knee_angle_right) / 2
        
        # Deteksi posisi tidur (horizontal)
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        body_orientation = abs(left_shoulder.y - left_hip.y)
        
        if body_orientation < 0.1 and shoulder_width > 0.2:
            return Activity.LYING
        
        # Deteksi posisi jongkok
        elif (avg_knee_angle < 90 and avg_hip_angle < 90 and 
              shoulder_hip_diff > 0.15 and avg_body_height < 0.7):
            return Activity.SQUATTING
        
        # Deteksi posisi duduk
        elif avg_hip_angle < 60 and avg_knee_angle < 130:
            return Activity.SITTING
        
        # Deteksi berjalan
        elif ankle_height_diff > 0.05 and avg_knee_angle > 120:
            return Activity.WALKING
        
        # Deteksi berdiri
        elif avg_hip_angle > 150 and avg_knee_angle > 160:
            return Activity.STANDING
        
        else:
            return Activity.UNKNOWN
    
    def process_frame(self, frame, prev_landmarks=None, time_delta=0.033):
        """Memproses frame dan mendeteksi aktivitas"""
        # Konversi BGR ke RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process dengan MediaPipe
        results = self.pose.process(rgb_frame)
        
        activity = Activity.UNKNOWN
        landmarks = None
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            activity = self.detect_activity(landmarks, prev_landmarks, time_delta)
            
            # Gambar landmarks pose
            self.mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
        
        return frame, activity, landmarks

def main():
    detector = RehabActivityDetector()
    cap = cv2.VideoCapture(0)  # Gunakan kamera 0
    
    # Variabel untuk tracking history
    activity_history = []
    max_history = 10
    prev_landmarks = None
    last_time = time.time()
    
    print("Tekan 'q' untuk keluar")
    print("Deteksi Aktivitas Rehab: Duduk, Berdiri, Berjalan, Tidur, Jongkok, JATUH")
    print("ALARM: Akan berbunyi jika terdeteksi jatuh!")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontal untuk mirror effect
        frame = cv2.flip(frame, 1)
        
        # Hitung time delta
        current_time = time.time()
        time_delta = current_time - last_time
        last_time = current_time
        
        # Process frame
        processed_frame, activity, landmarks = detector.process_frame(frame, prev_landmarks, time_delta)
        
        # Update activity history
        activity_history.append(activity)
        if len(activity_history) > max_history:
            activity_history.pop(0)
        
        # Tentukan aktivitas dominan
        if activity_history:
            dominant_activity = max(set(activity_history), key=activity_history.count)
        else:
            dominant_activity = Activity.UNKNOWN
        
        # Tampilkan informasi aktivitas dengan warna berbeda untuk jatuh
        text_color = (0, 255, 0)  # Hijau default
        if dominant_activity == Activity.FALLING:
            text_color = (0, 0, 255)  # Merah untuk jatuh
            # Tambahkan background merah untuk peringatan
            cv2.rectangle(processed_frame, (5, 5), (400, 40), (0, 0, 255), -1)
            cv2.putText(processed_frame, "⛔ PERINGATAN: JATUH TERDETEKSI! ⛔", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(processed_frame, f"Aktivitas: {dominant_activity.value}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        
        # Tampilkan status alarm
        alarm_status = "ALARM AKTIF!" if detector.alarm_manager.is_playing else "Alarm siap"
        alarm_color = (0, 0, 255) if detector.alarm_manager.is_playing else (0, 255, 0)
        cv2.putText(processed_frame, f"Status: {alarm_status}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, alarm_color, 2)
        
        # Tampilkan detail sudut jika landmarks terdeteksi
        if landmarks:
            left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
            left_knee = landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value]
            right_knee = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value]
            
            hip_angle_left = detector.calculate_angle(
                landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value],
                left_hip,
                left_knee
            )
            knee_angle_left = detector.calculate_angle(left_hip, left_knee,
                landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value])
            
            y_pos = 100
            cv2.putText(processed_frame, f"Sudut Pinggul: {hip_angle_left:.1f} derajat", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(processed_frame, f"Sudut Lutut: {knee_angle_left:.1f} derajat", 
                       (10, y_pos + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Tampilkan instruksi
        cv2.putText(processed_frame, "Tekan 'q' untuk keluar", 
                   (10, processed_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Tampilkan frame
        cv2.imshow('Deteksi Aktivitas Rehab + Jatuh', processed_frame)
        
        # Update previous landmarks untuk deteksi kecepatan
        prev_landmarks = landmarks
        
        # Keluar jika tombol 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Hentikan alarm sebelum keluar
            detector.alarm_manager.stop_alarm()
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
