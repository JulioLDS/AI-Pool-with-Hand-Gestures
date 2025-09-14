import cv2
import mediapipe as mp

# Inicializa o MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Abre a câmera
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,  # número máximo de mãos
    min_detection_confidence=0.7,  # confiança mínima
    min_tracking_confidence=0.7
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Espelha para ficar mais natural
        frame = cv2.flip(frame, 1)

        # Converte para RGB (MediaPipe usa RGB, OpenCV usa BGR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Processa a imagem e detecta mãos
        results = hands.process(rgb_frame)

        # Se encontrar alguma mão
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Desenha os pontos e conexões na mão
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(132, 0, 255), thickness=2, circle_radius=3),  # pontos
                    mp_drawing.DrawingSpec(color=(80, 255, 138), thickness=2)  # conexões
                )


        # Mostra o resultado
        cv2.imshow("Detecção de Mão", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
