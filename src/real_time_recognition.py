import cv2
import joblib
from hand_detection import HandDetector

# Carrega o modelo treinado
model = joblib.load('models/hand_model.pkl')

# Inicializa o detector de mãos
detector = HandDetector()

# Configura a captura de vídeo
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Espelha a imagem
    frame = cv2.flip(frame, 1)

    # Detecta as mãos e landmarks
    frame, hand_landmarks = detector.detect_hands(frame)

    # Se houver landmarks, faz a previsão
    if hand_landmarks:
        for hand_lms in hand_landmarks:
            # Extrai os landmarks (21 pontos x, y, z)
            landmarks = detector.extract_landmarks(hand_lms)
            landmarks_flatten = [coord for landmark in landmarks for coord in landmark]  # Achata a lista para 1D

            # Transforma a lista em um array 2D (1 amostra x 63 características)
            landmarks_flatten = [landmarks_flatten]

            # Faz a previsão
            prediction = model.predict(landmarks_flatten)
            print(f"Letra detectada: {prediction[0]}")

            # Adiciona a previsão na imagem
            cv2.putText(frame, f"Letra: {prediction[0]}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Mostra a imagem
    cv2.imshow('Reconhecimento de Letras', frame)

    # Pressiona 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
