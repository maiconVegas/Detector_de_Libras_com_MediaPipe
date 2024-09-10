import cv2
import csv
import os
from hand_detection import HandDetector

# Configuração inicial
detector = HandDetector()
cap = cv2.VideoCapture(0)

# Nome do arquivo para armazenar os dados
output_file = 'dataset/hand_landmarks.csv'

# Se o arquivo não existir, cria-o com o cabeçalho
if not os.path.exists(output_file):
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # O cabeçalho será: ['label', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', ..., 'x21', 'y21', 'z21']
        header = ['label']
        for i in range(21):
            header += [f'x{i+1}', f'y{i+1}', f'z{i+1}']
        writer.writerow(header)

print("Iniciando coleta de dados. Pressione a tecla correspondente para salvar a letra (A-Z) ou 'q' para sair.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Erro na captura do vídeo")
        break

    # Espelha a imagem
    frame = cv2.flip(frame, 1)

    # Detecta as mãos
    frame, hand_landmarks = detector.detect_hands(frame)

    # Mostra a imagem com as marcações
    cv2.imshow('Coleta de Dados de Libras', frame)

    # Captura a tecla pressionada
    key = cv2.waitKey(1) & 0xFF

    if hand_landmarks:
        for hand_lms in hand_landmarks:
            landmarks = detector.extract_landmarks(hand_lms)

            # Verifica se foi pressionada uma tecla correspondente a uma letra
            if key in range(ord('a'), ord('z') + 1):
                letter = chr(key).upper()  # Converte para letra maiúscula
                print(f"Capturando dados para a letra: {letter}")

                # Salva os landmarks no arquivo CSV
                with open(output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    row = [letter] + [coord for landmark in landmarks for coord in landmark]
                    writer.writerow(row)

            # Se a tecla 'q' for pressionada, sai do loop
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
