import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, max_num_hands=2, detection_confidence=0.7, tracking_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=max_num_hands,
                                         min_detection_confidence=detection_confidence,
                                         min_tracking_confidence=tracking_confidence)
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_hands(self, frame):
        # Converte para RGB para o MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Processa a imagem para detectar as mãos
        results = self.hands.process(image_rgb)
        
        # Converte de volta para BGR para exibição
        frame.flags.writeable = True
        frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Verifica se as mãos foram detectadas
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Desenha os landmarks e as conexões
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
        return frame, results.multi_hand_landmarks

    def extract_landmarks(self, hand_landmarks):
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])
        return landmarks
