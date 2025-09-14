import cv2
import mediapipe as mp
import math
import collections
import time

# --------------------
# Configurações (ajuste à vontade)
# --------------------
TABLE_MARGIN_X = 60      # margem horizontal (px)
TABLE_MARGIN_Y = 80      # margem vertical (px)
BALL_RADIUS = 30         # raio da bola (px)
FRICTION = 0.995        # fricção por frame (1 = sem fricção)
RESTITUTION = 0.90      # perda de energia ao ricochetear (0..1)
HISTORY_LEN = 5         # quantos pontos usar para estimar movimento do dedo
PUSH_BASE_SPEED = 10.0  # velocidade base ao empurrar (px por frame)
PUSH_SPEED_MULT = 1.6   # multiplica a média de movimento do dedo
TOUCH_COOLDOWN = 0.15   # segundos entre empurrões consecutivos

# --------------------
# Funções auxiliares
# --------------------
def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def is_pointing(hand_landmarks):
    """Retorna True se apenas o indicador estiver levantado."""
    index_tip = hand_landmarks.landmark[8]
    index_pip = hand_landmarks.landmark[6]

    middle_tip = hand_landmarks.landmark[12]
    middle_pip = hand_landmarks.landmark[10]

    ring_tip = hand_landmarks.landmark[16]
    ring_pip = hand_landmarks.landmark[14]

    pinky_tip = hand_landmarks.landmark[20]
    pinky_pip = hand_landmarks.landmark[18]

    index_up = index_tip.y < index_pip.y
    middle_down = middle_tip.y > middle_pip.y
    ring_down = ring_tip.y > ring_pip.y
    pinky_down = pinky_tip.y > pinky_pip.y

    return index_up and middle_down and ring_down and pinky_down

# --------------------
# Inicialização
# --------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

index_history = collections.deque(maxlen=HISTORY_LEN)  # guarda (x_px, y_px, t)
last_push_time = 0.0

# pos/vel da bola (serão inicializadas quando soubermos w,h)
ball_x = None
ball_y = None
ball_vx = 0.0
ball_vy = 0.0

with mp_hands.Hands(max_num_hands=1,
                    min_detection_confidence=0.6,
                    min_tracking_confidence=0.6) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # calcula a área da "mesa" (retângulo interno)
        left = TABLE_MARGIN_X
        top = TABLE_MARGIN_Y
        right = w - TABLE_MARGIN_X
        bottom = h - TABLE_MARGIN_Y

        # inicializa bola centrada na mesa (na primeira iteração)
        if ball_x is None:
            ball_x = (left + right) / 2.0
            ball_y = (top + bottom) / 2.0

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        index_found = False
        pointing = False
        avg_ix = avg_iy = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # posição do indicador (em pixels)
                index_tip = hand_landmarks.landmark[8]
                ix = int(index_tip.x * w)
                iy = int(index_tip.y * h)
                now = time.time()

                # atualiza histórico do dedo
                index_history.append((ix, iy, now))
                index_found = True

                # média do histórico para suavizar ruído
                sumx = sum(p[0] for p in index_history)
                sumy = sum(p[1] for p in index_history)
                avg_ix = int(round(sumx / len(index_history)))
                avg_iy = int(round(sumy / len(index_history)))

                # desenho do indicador suavizado
                cv2.circle(frame, (avg_ix, avg_iy), 8, (0, 255, 0), -1)

                # detectar gesto (apenas indicador levantado)
                pointing = is_pointing(hand_landmarks)

                # se está apontando e tocando a bola -> empurrar (com cooldown)
                if pointing:
                    # distância entre dedo (médio suavizado) e centro da bola
                    dist = math.hypot(avg_ix - ball_x, avg_iy - ball_y)
                    if dist <= BALL_RADIUS + 12:  # margem de toque
                        if now - last_push_time > TOUCH_COOLDOWN:
                            # direção do empurrão: do dedo para o centro da bola (bola sai para fora do dedo)
                            dx = ball_x - avg_ix
                            dy = ball_y - avg_iy
                            norm = math.hypot(dx, dy)
                            if norm == 0:
                                # fallback aleatório
                                push_dir_x, push_dir_y = 1.0, 0.0
                            else:
                                push_dir_x = dx / norm
                                push_dir_y = dy / norm

                            # estimativa do movimento médio do dedo em px/quadros
                            if len(index_history) >= 2:
                                oldest = index_history[0]
                                newest = index_history[-1]
                                frames = max(1, len(index_history) - 1)
                                avg_move = math.hypot(newest[0] - oldest[0], newest[1] - oldest[1]) / frames
                            else:
                                avg_move = 0.0

                            # velocidade resultante = base + movimento do dedo * multiplicador
                            speed = PUSH_BASE_SPEED + avg_move * PUSH_SPEED_MULT

                            # aplica velocidade (px por frame)
                            ball_vx = push_dir_x * speed
                            ball_vy = push_dir_y * speed

                            last_push_time = now

        else:
            # se não detectou dedo, limpar histórico (evita usar valores antigos)
            index_history.clear()

        # --------------------
        # Atualiza física da bola
        # --------------------
        ball_x += ball_vx
        ball_y += ball_vy

        # fricção (leva a bola a parar lentamente)
        ball_vx *= FRICTION
        ball_vy *= FRICTION

        # ricochete correto com reposicionamento (para evitar "colar" na borda)
        # esquerda
        if ball_x - BALL_RADIUS < left:
            ball_x = left + BALL_RADIUS
            ball_vx = abs(ball_vx) * RESTITUTION
        # direita
        if ball_x + BALL_RADIUS > right:
            ball_x = right - BALL_RADIUS
            ball_vx = -abs(ball_vx) * RESTITUTION
        # topo
        if ball_y - BALL_RADIUS < top:
            ball_y = top + BALL_RADIUS
            ball_vy = abs(ball_vy) * RESTITUTION
        # baixo
        if ball_y + BALL_RADIUS > bottom:
            ball_y = bottom - BALL_RADIUS
            ball_vy = -abs(ball_vy) * RESTITUTION

        # --------------------
        # Desenhos na tela
        # --------------------
        # desenha a "mesa" (retângulo)
        cv2.rectangle(frame, (left, top), (right, bottom), (30, 120, 30), 6)  # cor verde escura da mesa

        # desenha bola (converte pra int)
        cv2.circle(frame, (int(ball_x), int(ball_y)), BALL_RADIUS, (200, 30, 30), -1)

        # HUD simples
        status = "Pointing" if pointing else "No pointing"
        cv2.putText(frame, f"{status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # desenha uma linha guia do dedo até a bola (opcional, ajuda a ver direção)
        if index_found and avg_ix is not None:
            cv2.line(frame, (avg_ix, avg_iy), (int(ball_x), int(ball_y)), (180, 180, 180), 1)

        cv2.imshow("Mesa de Bilhar (gestos)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
