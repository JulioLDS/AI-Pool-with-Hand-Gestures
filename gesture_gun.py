import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def distance(a, b):
    """ Calcula a distância euclidiana entre dois pontos (x,y) """
    return math.hypot(a.x - b.x, a.y - b.y)

def is_gun_gesture(hand_landmarks):
    """
    Retorna True se a mão estiver no gesto de 'arma'
    (indicador e polegar esticados, outros dobrados)
    """

    # Pontos importantes
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    thumb_mcp = hand_landmarks.landmark[2]

    index_tip = hand_landmarks.landmark[8]
    index_pip = hand_landmarks.landmark[6]

    middle_tip = hand_landmarks.landmark[12]
    middle_pip = hand_landmarks.landmark[10]

    ring_tip = hand_landmarks.landmark[16]
    ring_pip = hand_landmarks.landmark[14]

    pinky_tip = hand_landmarks.landmark[20]
    pinky_pip = hand_landmarks.landmark[18]

    # Regras usando distância:
    thumb_extended = distance(thumb_tip, thumb_mcp) > 0.1
    index_extended = index_tip.y < index_pip.y
    middle_folded = middle_tip.y > middle_pip.y
    ring_folded = ring_tip.y > ring_pip.y
    pinky_folded = pinky_tip.y > pinky_pip.y

    return (thumb_extended and index_extended and
            middle_folded and ring_folded and pinky_folded)



cap = cv2.VideoCapture(0)
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if is_gun_gesture(hand_landmarks):
                    cv2.putText(frame, "GESTO DE ARMA DETECTADO!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow("Deteccao de Gesto", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
