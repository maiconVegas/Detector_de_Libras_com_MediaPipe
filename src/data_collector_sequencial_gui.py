import cv2
import tkinter as tk
from tkinter import ttk
import threading
import time
from hand_detection import HandDetector
import csv
import os

class DataCollectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Coletor de Dados de Libras")

        # Variáveis
        self.recording = False
        self.start_time = None
        self.dataset_file = 'dataset/hand_landmarks.csv'
        self.detector = HandDetector()
        self.frame_sequence = []
        self.max_frames = 30  # Número máximo de frames para coletar (aproximadamente 2 segundos a 15 FPS)

        # Layout
        self.label = tk.Label(root, text="Digite a letra ou palavra:")
        self.label.pack(pady=10)

        self.text_entry = tk.Entry(root)
        self.text_entry.pack(pady=5)

        self.start_button = ttk.Button(root, text="Iniciar Coleta", command=self.start_recording)
        self.start_button.pack(pady=10)

        self.stop_button = ttk.Button(root, text="Parar Coleta", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(pady=10)

        self.quit_button = ttk.Button(root, text="Fechar", command=root.quit)
        self.quit_button.pack(pady=10)

        # Captura de vídeo
        self.cap = cv2.VideoCapture(0)
        self.thread = threading.Thread(target=self.process_video)
        self.thread.start()

    def start_recording(self):
        self.recording = True
        self.start_time = time.time()
        self.frame_sequence = []
        self.stop_button.config(state=tk.NORMAL)
        self.start_button.config(state=tk.DISABLED)
        print("Iniciando coleta de dados...")

    def stop_recording(self):
        self.recording = False
        self.stop_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.NORMAL)
        if self.frame_sequence:
            print("Coleta de dados encerrada.")
            self.save_data()
        else:
            print("Nenhum dado coletado.")

    def process_video(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Espelha a imagem
            frame, hand_landmarks = self.detector.detect_hands(frame)

            if self.recording:
                self.frame_sequence.append(frame.copy())  # Armazena uma cópia do frame
                elapsed_time = time.time() - self.start_time
                if elapsed_time > 2:  # Tempo total de coleta
                    self.stop_recording()

            cv2.imshow('Coletor de Dados', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def save_data(self):
        letter = self.text_entry.get().upper() if self.text_entry.get() else "UNKNOWN"
        for i, frame in enumerate(self.frame_sequence):
            # Extrai os landmarks
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hand_landmarks = self.detector.detect_hands(frame)[1]
            if hand_landmarks:
                landmarks = self.detector.extract_landmarks(hand_landmarks[0])
                landmarks_flatten = [coord for landmark in landmarks for coord in landmark]
                landmarks_flatten = [landmarks_flatten]

                if not os.path.exists(self.dataset_file):
                    with open(self.dataset_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        header = ['label']
                        for i in range(21):
                            header += [f'x{i+1}', f'y{i+1}', f'z{i+1}']
                        writer.writerow(header)

                with open(self.dataset_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    row = [letter] + landmarks_flatten[0]
                    writer.writerow(row)

if __name__ == "__main__":
    root = tk.Tk()
    app = DataCollectorApp(root)
    root.mainloop()
