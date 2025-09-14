import cv2

# Abre a câmera (0 = webcam padrão)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Não foi possível acessar a câmera.")
    exit()

while True:
    # Lê um frame da câmera
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar frame.")
        break

    # Mostra o frame
    frame = cv2.flip(frame, 1)  # espelha horizontalmente
    cv2.imshow("Teste da Câmera", frame)

    # Sai se apertar a tecla "q"
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libera recursos
cap.release()
cv2.destroyAllWindows()
