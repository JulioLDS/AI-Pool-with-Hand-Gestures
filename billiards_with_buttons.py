import cv2
import mediapipe as mp
import math
import collections
import time
import random
import os

# --------------------
# Configurações
# --------------------
TABLE_MARGIN_X = 60
TABLE_MARGIN_Y = 80
BALL_RADIUS = 10
FRICTION = 0.992
RESTITUTION = 0.90
COL_RESTITUTION = 0.95

HISTORY_LEN = 7
POINTING_WINDOW = 6
POINTING_MIN_TRUE = 4
NO_DET_GRACE = 0.25

PUSH_BASE_SPEED = 8.0
PUSH_SPEED_MULT = 1.3
TOUCH_COOLDOWN = 0.15
POCKET_RADIUS = 28

# --------------------
# Funções utilitárias
# --------------------
def distance_xy(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)

def is_pointing(hand_landmarks):
    """
    Retorna True se apenas o indicador estiver levantado.
    Usa distâncias normalizadas para maior robustez.
    """
    lt = hand_landmarks.landmark

    def finger_extended(tip, pip, mcp):
        return (lt[tip].y < lt[pip].y) and (lt[pip].y < lt[mcp].y)

    # Indicador levantado
    index_up = finger_extended(8, 6, 5)

    # Outros dedos dobrados
    middle_down = lt[12].y > lt[10].y
    ring_down = lt[16].y > lt[14].y
    pinky_down = lt[20].y > lt[18].y

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

    def update(self, left, top, right, bottom, pockets):
        """
        Atualiza posição, aplica fricção, trata ricochete e checa caçapa.
        pockets: lista de tuplas (px,py)
        """
        if not self.alive:
            return

        # Atualiza posição com velocidade atual
        self.x += self.vx
        self.y += self.vy

        # aplicar atrito
        self.vx *= FRICTION
        self.vy *= FRICTION

        # evitar velocidade residual que trava a bola
        if abs(self.vx) < 0.05:
            self.vx = 0.0
        if abs(self.vy) < 0.05:
            self.vy = 0.0

        # colisão com bordas (ricochete) — reposiciona para evitar "colar"
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

        # verificar se caiu na caçapa (apenas bolas não-brancas desaparecem)
        for px, py in pockets:
            if distance_xy(self.x, self.y, px, py) < POCKET_RADIUS:
                if not self.is_white:
                    self.alive = False
                # Se for a branca, não removemos — mantemos em jogo (pode ajustar se preferir)
                break

    def draw(self, frame):
        if self.alive:
            cv2.circle(frame, (int(self.x), int(self.y)), BALL_RADIUS, self.color, -1)

def handle_ball_collision(b1, b2):
    """
    Colisão elástica simplificada entre duas bolas (mesma massa).
    """
    if not b1.alive or not b2.alive:
        return
    dx = b2.x - b1.x
    dy = b2.y - b1.y
    dist = math.hypot(dx, dy)
    if dist <= 0:
        return
    min_dist = 2 * BALL_RADIUS
    if dist < min_dist:
        nx = dx / dist
        ny = dy / dist
        overlap = (min_dist - dist)
        # separar bolas
        b1.x -= nx * overlap / 2
        b1.y -= ny * overlap / 2
        b2.x += nx * overlap / 2
        b2.y += ny * overlap / 2

        # vetores tangente/normal
        tx = -ny
        ty = nx

        # componentes de velocidade
        v1n = b1.vx * nx + b1.vy * ny
        v1t = b1.vx * tx + b1.vy * ty
        v2n = b2.vx * nx + b2.vy * ny
        v2t = b2.vx * tx + b2.vy * ty

        # trocam as componentes normais (massas iguais)
        v1n, v2n = v2n, v1n

        # reconstruir velocidades
        b1.vx = v1n * nx + v1t * tx
        b1.vy = v1n * ny + v1t * ty
        b2.vx = v2n * nx + v2t * tx
        b2.vy = v2n * ny + v2t * ty

        # pequena perda de energia
        b1.vx *= COL_RESTITUTION
        b1.vy *= COL_RESTITUTION
        b2.vx *= COL_RESTITUTION
        b2.vy *= COL_RESTITUTION

# --------------------
# Funções de jogo
# --------------------
def reset_balls(left, top, right, bottom):
    balls = []

    # Bola branca no centro da mesa
    white_x = (left + right) / 2
    white_y = (top + bottom) / 2
    balls.append(Ball(white_x, white_y, (255, 255, 255), is_white=True))

    # Cores das 15 bolas
    colors = [
        (200, 30, 30), (30, 200, 30), (30, 30, 200),
        (200, 200, 30), (200, 30, 200), (30, 200, 200),
        (160, 100, 30), (30, 160, 160),
        (255, 100, 100), (100, 255, 100), (100, 100, 255),
        (255, 255, 100), (255, 100, 255), (100, 255, 255),
        (180, 180, 180)
    ]

    # Triângulo virado para a esquerda, afastado um pouco da borda
    spacing = BALL_RADIUS * 2 + 2
    start_x = left + 140
    start_y = (top + bottom) / 2

    k = 0
    for row in range(5):  # 5 fileiras
        for col in range(row + 1):
            if k >= len(colors):
                break
            x = start_x - row * spacing
            y = start_y - row * spacing / 2 + col * spacing
            balls.append(Ball(x, y, colors[k]))
            k += 1

    return balls

def draw_button(frame, text, center, size=(160, 60)):
    cx, cy = center
    w, h = size
    left, top = cx - w//2, cy - h//2
    right, bottom = cx + w//2, cy + h//2
    cv2.rectangle(frame, (left, top), (right, bottom), (50, 50, 200), -1)
    cv2.putText(frame, text, (left+20, cy+8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    return (left, top, right, bottom)

def point_in_rect(x, y, rect):
    left, top, right, bottom = rect
    return left <= x <= right and top <= y <= bottom

# --------------------
# Main Loop
# --------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

index_history = collections.deque(maxlen=HISTORY_LEN)
pointing_hist = collections.deque(maxlen=POINTING_WINDOW)
last_push_time = 0.0
last_seen_time = 0.0

game_state = "menu"  # "menu", "playing", "gameover"
balls = []

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,   # antes 0.7
    min_tracking_confidence=0.6     # antes 0.7
) as hands:


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        left, top, right, bottom = TABLE_MARGIN_X, TABLE_MARGIN_Y, w - TABLE_MARGIN_X, h - TABLE_MARGIN_Y
        cx = (left + right) // 2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        pointing = False
        avg_ix = avg_iy = None
        now = time.time()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_tip = hand_landmarks.landmark[8]
                ix = int(index_tip.x * w)
                iy = int(index_tip.y * h)
                index_history.append((ix, iy, now))
                avg_ix = int(sum(p[0] for p in index_history) / len(index_history))
                avg_iy = int(sum(p[1] for p in index_history) / len(index_history))
                cv2.circle(frame, (avg_ix, avg_iy), 8, (0, 255, 0), -1)

                pointing_now = is_pointing(hand_landmarks)
                pointing_hist.append(pointing_now)
                last_seen_time = now
                pointing = sum(1 for v in pointing_hist if v) >= POINTING_MIN_TRUE
        else:
            if now - last_seen_time > NO_DET_GRACE:
                index_history.clear()
                pointing_hist.clear()

        # --------------------
        # MENU
        # --------------------
        if game_state == "menu":
            cv2.putText(frame, "Bilhar com Gestos", (w//2 - 160, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)
            btn_rect = draw_button(frame, "  START ", (w//2, h//2))
            if pointing and avg_ix and point_in_rect(avg_ix, avg_iy, btn_rect):
                balls = reset_balls(left, top, right, bottom)
                game_state = "playing"

        # --------------------          
        # PLAYING
        # --------------------
        elif game_state == "playing":
            # Atualizar e desenhar mesa
            cv2.rectangle(frame, (left, top), (right, bottom), (30, 120, 30), 6)
            pockets = [(left, top), (cx, top), (right, top),
                       (left, bottom), (cx, bottom), (right, bottom)]
            for px, py in pockets:
                cv2.circle(frame, (px, py), POCKET_RADIUS, (0, 0, 0), -1)

            # Empurrar bola branca
            if pointing and avg_ix:
                white_ball = [b for b in balls if b.is_white][0]
                dist = distance_xy(avg_ix, avg_iy, white_ball.x, white_ball.y)
                if dist <= BALL_RADIUS + 12 and now - last_push_time > TOUCH_COOLDOWN:
                    dx = white_ball.x - avg_ix
                    dy = white_ball.y - avg_iy
                    norm = math.hypot(dx, dy)
                    if norm != 0:
                        push_dir_x = dx / norm
                        push_dir_y = dy / norm
                        if len(index_history) >= 2:
                            oldest = index_history[0]
                            newest = index_history[-1]
                            frames = max(1, len(index_history)-1)
                            avg_move = math.hypot(newest[0]-oldest[0], newest[1]-oldest[1]) / frames
                        else:
                            avg_move = 0
                        speed = PUSH_BASE_SPEED + avg_move * PUSH_SPEED_MULT
                        white_ball.vx = push_dir_x * speed
                        white_ball.vy = push_dir_y * speed
                        last_push_time = now

            # Atualiza bolas (agora passando pockets)
            for b in balls:
                b.update(left, top, right, bottom, pockets)

            # Colisões
            for i in range(len(balls)):
                for j in range(i+1, len(balls)):
                    handle_ball_collision(balls[i], balls[j])

            # Caçapas (já tratado em update; mantemos checagem para lógica adicional)
            for b in balls:
                if not b.alive: continue
                for px, py in pockets:
                    if distance_xy(b.x, b.y, px, py) < POCKET_RADIUS:
                        b.alive = False
                        break

            # Desenhar bolas
            for b in balls:
                b.draw(frame)

            # Condição de fim de jogo
            white_alive = any(b.is_white and b.alive for b in balls)
            any_color_alive = any((not b.is_white) and b.alive for b in balls)
            if not white_alive or not any_color_alive:
                game_state = "gameover"

        # --------------------
        # GAME OVER
        # --------------------
        elif game_state == "gameover":
            msg = "VOCE GANHOU!" if any(b.is_white and b.alive for b in balls) else "VOCE PERDEU!"
            cv2.putText(frame, msg, (w//2 - 120, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3)
            btn_rect = draw_button(frame, "RESTART", (w//2, h//2))
            if pointing and avg_ix and point_in_rect(avg_ix, avg_iy, btn_rect):
                balls = reset_balls(left, top, right, bottom)
                game_state = "playing"

        cv2.imshow("Bilhar com Gestos", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
