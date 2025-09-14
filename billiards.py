import cv2
import mediapipe as mp
import math
import collections
import time
import random

# --------------------
# Configurações da mesa e física
# --------------------
TABLE_MARGIN_X = 60
TABLE_MARGIN_Y = 80
BALL_RADIUS = 15          # bolas menores
FRICTION = 0.992          # atrito por frame
RESTITUTION = 0.90        # perda em ricochete em tabelas
COL_RESTITUTION = 0.95    # perda em colisão entre bolas

HISTORY_LEN = 7           # suavização do indicador
POINTING_WINDOW = 6       # janela temporal do gesto
POINTING_MIN_TRUE = 4     # precisa ser True em >= 4/6
NO_DET_GRACE = 0.25       # seg. de "carência" sem mão antes de limpar histórico

PUSH_BASE_SPEED = 8.0
PUSH_SPEED_MULT = 1.3
TOUCH_COOLDOWN = 0.15     # seg. entre empurrões

# Caçapas (cantos + meio das bordas horizontais)
POCKET_RADIUS = 28

# --------------------
# Utilidades
# --------------------
def distance_xy(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)

def is_pointing_relaxed(hand_landmarks):
    """
    Gesto de "apontar" mais tolerante:
    - Indicador: tip acima do PIP com folga
    - Demais (médio, anelar, mínimo): tip abaixo do PIP (com pequena tolerância)
    Usa coordenadas normalizadas do MediaPipe (0..1, eixo Y cresce pra baixo).
    """
    # Índices dos pontos (MediaPipe)
    idx_tip, idx_pip = 8, 6
    mid_tip, mid_pip = 12, 10
    rng_tip, rng_pip = 16, 14
    pky_tip, pky_pip = 20, 18

    lt = hand_landmarks.landmark
    # folgas em unidades normalizadas (≈ 2% da altura da imagem)
    up_eps = 0.02      # quanto o indicador deve estar "acima"
    down_eps = 0.005   # tolerância para considerar dedo "abaixo"

    index_up = lt[idx_tip].y < lt[idx_pip].y - up_eps
    middle_down = lt[mid_tip].y > lt[mid_pip].y - down_eps
    ring_down = lt[rng_tip].y > lt[rng_pip].y - down_eps
    pinky_down = lt[pky_tip].y > lt[pky_pip].y - down_eps

    return index_up and middle_down and ring_down and pinky_down

# --------------------
# Classe Bola
# --------------------
class Ball:
    def __init__(self, x, y, color, is_white=False):
        self.x = float(x)
        self.y = float(y)
        self.vx = 0.0
        self.vy = 0.0
        self.color = color
        self.is_white = is_white
        self.alive = True

    def update(self, left, top, right, bottom):
        if not self.alive:
            return
        # Atualiza posição
        self.x += self.vx
        self.y += self.vy

        # Atrito
        self.vx *= FRICTION
        self.vy *= FRICTION

        # Ricochetes nas tabelas
        if self.x - BALL_RADIUS < left:
            self.x = left + BALL_RADIUS
            self.vx = abs(self.vx) * RESTITUTION
        if self.x + BALL_RADIUS > right:
            self.x = right - BALL_RADIUS
            self.vx = -abs(self.vx) * RESTITUTION
        if self.y - BALL_RADIUS < top:
            self.y = top + BALL_RADIUS
            self.vy = abs(self.vy) * RESTITUTION
        if self.y + BALL_RADIUS > bottom:
            self.y = bottom - BALL_RADIUS
            self.vy = -abs(self.vy) * RESTITUTION

    def draw(self, frame):
        if not self.alive:
            return
        cv2.circle(frame, (int(self.x), int(self.y)), BALL_RADIUS, self.color, -1)

# --------------------
# Colisão entre bolas (2D, massas iguais)
# --------------------
def handle_ball_collision(b1, b2):
    if not b1.alive or not b2.alive:
        return
    dx = b2.x - b1.x
    dy = b2.y - b1.y
    dist = math.hypot(dx, dy)
    if dist <= 0:
        return

    min_dist = 2 * BALL_RADIUS
    if dist < min_dist:
        # Vetor normal
        nx = dx / dist
        ny = dy / dist
        # Separação para evitar "afundar"
        overlap = (min_dist - dist)
        b1.x -= nx * overlap / 2
        b1.y -= ny * overlap / 2
        b2.x += nx * overlap / 2
        b2.y += ny * overlap / 2

        # Vetor tangente
        tx = -ny
        ty = nx

        # Decompõe velocidades
        v1n = b1.vx * nx + b1.vy * ny
        v1t = b1.vx * tx + b1.vy * ty
        v2n = b2.vx * nx + b2.vy * ny
        v2t = b2.vx * tx + b2.vy * ty

        # Colisão elástica para a componente normal (massas iguais: trocam normales)
        v1n, v2n = v2n, v1n

        # Reconstrói as velocidades
        b1.vx = v1n * nx + v1t * tx
        b1.vy = v1n * ny + v1t * ty
        b2.vx = v2n * nx + v2t * tx
        b2.vy = v2n * ny + v2t * ty

        # Pequena perda de energia na colisão
        b1.vx *= COL_RESTITUTION
        b1.vy *= COL_RESTITUTION
        b2.vx *= COL_RESTITUTION
        b2.vy *= COL_RESTITUTION

# --------------------
# Inicialização de visão
# --------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

index_history = collections.deque(maxlen=HISTORY_LEN)      # (x, y, t)
pointing_hist = collections.deque(maxlen=POINTING_WINDOW)  # booleans
last_push_time = 0.0
last_seen_time = 0.0

balls = []

with mp_hands.Hands(max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Limites da mesa
        left  = TABLE_MARGIN_X
        top   = TABLE_MARGIN_Y
        right = w - TABLE_MARGIN_X
        bottom= h - TABLE_MARGIN_Y
        cx = (left + right) // 2  # centro X (para caçapas do meio)

        # Inicializa bolas uma única vez
        if not balls:
            # bola branca no centro
            balls.append(Ball((left + right) / 2, (top + bottom) / 2, (255, 255, 255), is_white=True))
            # rack simples de bolas coloridas
            colors = [(200, 30, 30), (30, 200, 30), (30, 30, 200),
                      (200, 200, 30), (200, 30, 200), (30, 200, 200),
                      (160, 100, 30), (30, 160, 160)]
            # posiciona em grade leve para evitar sobreposição inicial
            cols = 4
            spacing = BALL_RADIUS * 2 + 6
            start_x = left + spacing*2
            start_y = top + spacing*2
            k = 0
            for r in range(2):
                for c in range(cols):
                    if k >= len(colors): break
                    x = start_x + c*spacing
                    y = start_y + r*spacing
                    balls.append(Ball(x, y, colors[k]))
                    k += 1

        # MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        pointing_now = False
        avg_ix = avg_iy = None
        now = time.time()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Indicador (em px)
                index_tip = hand_landmarks.landmark[8]
                ix = int(index_tip.x * w)
                iy = int(index_tip.y * h)

                # Atualiza histórico e média (suavização)
                index_history.append((ix, iy, now))
                avg_ix = int(sum(p[0] for p in index_history) / len(index_history))
                avg_iy = int(sum(p[1] for p in index_history) / len(index_history))
                cv2.circle(frame, (avg_ix, avg_iy), 8, (0, 255, 0), -1)

                # Gesto (relaxado) + janela temporal
                pointing_now = is_pointing_relaxed(hand_landmarks)
                pointing_hist.append(pointing_now)
                last_seen_time = now

                # Considera "apontando" apenas se tiver maioria na janela
                pointing = sum(1 for v in pointing_hist if v) >= POINTING_MIN_TRUE

                # Só empurra a bola branca
                if pointing and avg_ix is not None:
                    white_ball = [b for b in balls if b.is_white][0]
                    dist = distance_xy(avg_ix, avg_iy, white_ball.x, white_ball.y)
                    if dist <= BALL_RADIUS + 12 and now - last_push_time > TOUCH_COOLDOWN:
                        dx = white_ball.x - avg_ix
                        dy = white_ball.y - avg_iy
                        norm = math.hypot(dx, dy)
                        if norm != 0:
                            push_dir_x = dx / norm
                            push_dir_y = dy / norm
                        else:
                            push_dir_x, push_dir_y = 1.0, 0.0

                        # estimativa de "força" do dedo
                        if len(index_history) >= 2:
                            oldest = index_history[0]
                            newest = index_history[-1]
                            frames = max(1, len(index_history) - 1)
                            avg_move = math.hypot(newest[0] - oldest[0], newest[1] - oldest[1]) / frames
                        else:
                            avg_move = 0.0

                        speed = PUSH_BASE_SPEED + avg_move * PUSH_SPEED_MULT
                        white_ball.vx = push_dir_x * speed
                        white_ball.vy = push_dir_y * speed
                        last_push_time = now
        else:
            # se perder a mão, aguarda pequena carência antes de limpar
            if now - last_seen_time > NO_DET_GRACE:
                index_history.clear()
                pointing_hist.clear()

        # Atualiza bolas
        for b in balls:
            b.update(left, top, right, bottom)

        # Colisão entre bolas
        for i in range(len(balls)):
            for j in range(i + 1, len(balls)):
                handle_ball_collision(balls[i], balls[j])

        # Caçapas (6: 4 cantos + 2 meio)
        pockets = [
            (left, top), (cx, top), (right, top),
            (left, bottom), (cx, bottom), (right, bottom)
        ]

        # Remover bolas que caem
        for b in balls:
            if not b.alive:
                continue
            for px, py in pockets:
                if distance_xy(b.x, b.y, px, py) < POCKET_RADIUS:
                    b.alive = False
                    break

        # Desenho da mesa
        cv2.rectangle(frame, (left, top), (right, bottom), (30, 120, 30), 6)
        for px, py in pockets:
            cv2.circle(frame, (px, py), POCKET_RADIUS, (0, 0, 0), -1)

        # Desenha bolas
        for b in balls:
            b.draw(frame)

        # HUD simples
        msg = "Aponte para empurrar a bola branca"
        cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)

        cv2.imshow("Bilhar com Gestos — v2", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
