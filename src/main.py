import cv2
from hand_detection import HandDetector

# Inicializa a detecção de mãos
detector = HandDetector()

# Configura a captura de vídeo
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar o vídeo")
        break

    # Espelha a imagem para não ficar invertida
    frame = cv2.flip(frame, 1)

    # Detecta as mãos e retorna o frame com os landmarks desenhados
    frame, hand_landmarks = detector.detect_hands(frame)

    # Mostra a imagem com as marcações
    cv2.imshow('DETECTOR DE LIBRAS', frame)

    # Exibe os landmarks (opcional: você pode salvar isso ou usar como dado de treino)
    if hand_landmarks:
        for hand_lms in hand_landmarks:
            landmarks = detector.extract_landmarks(hand_lms)
            print(landmarks)  # Mostra os landmarks no terminal

    # Sai do loop ao apertar 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
