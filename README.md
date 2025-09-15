# 🎱 Bilhar com Gestos (AI Pool with Hand Gestures)


Projeto de sinuca interativa controlada por gestos das mãos com Python, OpenCV e MediaPipe. A bola branca é movimentada apontando com o indicador, simulando tacadas reais, enquanto as demais seguem uma física simples de colisões e caçapas.
---

## 📌 Seções principais deste repositório
- `billiards_with_buttons.py` — código principal do jogo (detecção de mão + física + UI)
- `test_camera.py`, `test_hand.py` — scripts auxiliares para testar câmera e MediaPipe
- `images/` — imagens usadas no README e demonstrações
- `requirements.txt` — bibliotecas necessárias
- `Poppins-Bold.ttf` (opcional) — fonte usada para os botões (se aplicável)

---

## ✨ Visual — telas principais

**Tela inicial (Start)** — toque com o indicador no botão `START` para iniciar.  
<div align="center">
  <img src="images/start.png" alt="Tela inicial - Start" width="520"/>
</div>

**Mesa ociosa (idle)** — mesa pronta para jogar, bolas em rack triangular.  
<div align="center">
  <img src="images/idle.png" alt="Mesa ociosa" width="520"/>
</div>

**Em jogo (playing)** — exemplo com a bola branca em movimento durante a tacada.  
<div align="center">
  <img src="images/playing.png" alt="Jogando" width="520"/>
</div>

**Game Over / Restart** — quando a branca cai (ou todas as coloridas encaçapam), toque em `RESTART`.  
<div align="center">
  <img src="images/restart.png" alt="Restart" width="520"/>
</div>

---

# 🛠️ Instruções passo a passo (Windows / Linux / macOS)

> **Requisitos**  
> - Python **3.10** ou **3.11** (recomendado; o MediaPipe não tem builds oficiais para versões mais novas em alguns ambientes)  
> - Pip e virtualenv (opcional, mas recomendado)  
> - Webcam funcionando

## 1) Clonar o repositório
```bash
git clone <URL-do-seu-repo>
cd AI-Pool-with-Hand-Gestures
```

## 2) Criar e ativar ambiente virtual (recomendado)

**Windows (PowerShell)**
```powershell
py -3.10 -m venv venv
.env\Scriptsctivate
```

**Linux / macOS**
```bash
python3.10 -m venv venv
source venv/bin/activate
```

## 3) Instalar dependências
```bash
pip install --upgrade pip
pip install opencv-python mediapipe pygame numpy pillow
```
Opcional: criar `requirements.txt` com essas libs e usar `pip install -r requirements.txt`.

## 4) Testes rápidos

Teste a câmera:
```bash
python test_camera.py
```

Teste a detecção de mão (apenas visual):
```bash
python test_hand.py
```

## 5) Rodar o jogo
```bash
python billiards_with_buttons.py
```

Aponte o indicador para o botão START para iniciar.

Aponte para a bola branca e faça um movimento rápido para empurrá-la.

Ao final (vitória ou derrota), a tela mostrará RESTART — aponte no botão para reiniciar.

---

## 🎮 Como jogar (controle por gestos)

- **Start:** aponte o indicador sobre o botão START e mantenha o gesto de apontar (apenas o indicador esticado).
- **Mirar:** movimente o dedo para apontar a direção.
- **Tacada:** com o indicador tocando/encostando na bola branca, faça um movimento rápido (empurrão).
- **Restart:** após fim de jogo, aponte para o botão RESTART.

**Dicas:**
- Fique em frente a um fundo não muito confuso (boa iluminação melhora a detecção).
- Mantenha a mão relativamente central na câmera para melhores landmarks.
- Se o gesto falhar, tente aproximar/afastar levemente a mão da câmera até que o detector fique estável.

