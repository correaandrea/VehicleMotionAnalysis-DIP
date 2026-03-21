import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import calcular_centroide, calcular_cinematica

# 1. CONFIGURACIÓN GENERAL Y RUTAS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'VideoPDI.mp4')


# 2. PARÁMETROS FÍSICOS Y DE MEDICIÓN
# Límites espaciales para el análisis cinemático (Eje X)
MARCADOR_A = 387
MARCADOR_B = 718

# Calibración de escala (Metros por Píxel)
LONGITUD_CARRO_PIXELES = 41.0
LONGITUD_CARRO_METROS = 4.07
FACTOR_ESCALA = LONGITUD_CARRO_METROS / LONGITUD_CARRO_PIXELES

# 3. VARIABLES GLOBALES DE ESTADO
pausar_video = False
mostrar_pixel = False
x_pixel, y_pixel = 0, 0

# Almacenamiento de datos cinemáticos
trayectoria = []
posiciones_x = []

# 4. FUNCIONES DE INTERACCIÓN (HERRAMIENTAS DE DEPURACIÓN)
def mouse_callback(event, x, y, flags, param):
    """
    Captura eventos del mouse para análisis interactivo de píxeles y control de flujo.
    """
    global pausar_video, mostrar_pixel, x_pixel, y_pixel
    if event == cv2.EVENT_LBUTTONDOWN:
        x_pixel, y_pixel = x, y
        mostrar_pixel = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        pausar_video = not pausar_video

# 5. INICIALIZACIÓN DE VIDEO Y ENTORNOS
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"[ERROR] No se pudo inicializar el flujo de video en: {VIDEO_PATH}")
    exit()

# Ajuste de proporción (Aspect Ratio)
ancho_visualizacion = 800 
aspect_ratio = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / cap.get(cv2.CAP_PROP_FRAME_WIDTH)
alto_visualizacion = int(ancho_visualizacion * aspect_ratio)

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"[INFO] Video cargado: {ancho_visualizacion}x{alto_visualizacion} a {fps} FPS")

cv2.namedWindow('Analisis Matematico')
cv2.setMouseCallback('Analisis Matematico', mouse_callback)

# Inicialización del modelo de Sustracción de Fondo MOG2 
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

# 6. BUCLE PRINCIPAL DE PROCESAMIENTO
while True:
    if not pausar_video:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            trayectoria.clear() 
            posiciones_x.clear()
            continue
        
        frame_resized = cv2.resize(frame, (ancho_visualizacion, alto_visualizacion))
        
    frame_display = frame_resized.copy()

    # Herramienta de depuración interactiva en pantalla
    if mostrar_pixel:
        temp_hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
        color_hsv = temp_hsv[y_pixel, x_pixel]
        cv2.putText(frame_display, f'Pos:({x_pixel},{y_pixel}) HSV:{color_hsv}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # FASE A: Segmentación y Morfología 
    mascara = fgbg.apply(frame_resized)

    # Filtros morfológicos para aislamiento del vehículo
    kernel_apertura = np.ones((5, 5), np.uint8)
    kernel_cierre = np.ones((15, 25), np.uint8) 
    
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel_apertura) # Reducción de ruido
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel_cierre)  # Relleno de oclusiones

    centroide, contorno = calcular_centroide(mascara)

    # FASE B: Rastreo y Adquisición de Datos
    if centroide is not None:
        cx, cy = centroide
        
        # Registro condicional basado en límites espaciales (Marcadores A y B)
        if MARCADOR_A <= cx <= MARCADOR_B and not pausar_video:
            trayectoria.append((cx, cy))
            posiciones_x.append(cx)
            if len(trayectoria) > 120:
                trayectoria.pop(0)

        cv2.circle(frame_display, (cx, cy), 6, (0, 0, 255), -1)
        cv2.drawContours(frame_display, [contorno], -1, (0, 255, 0), 2)

    # Dibujo de la trayectoria histórica 
    for i in range(1, len(trayectoria)):
        cv2.line(frame_display, trayectoria[i-1], trayectoria[i], (255, 0, 0), 2)

    # FASE C: Cálculo Cinemático en Tiempo Real 
    if len(posiciones_x) >= 3:
        pos_m, vel, acel = calcular_cinematica(posiciones_x, fps, FACTOR_ESCALA)
        if len(vel) > 0:
            vel_kmh = vel[-1] * 3.6 
            texto_vel = f"Velocidad: {vel_kmh:.1f} km/h"
            cv2.putText(frame_display, texto_vel, (30, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # FASE D: Renderizado
    cv2.imshow('Analisis Matematico', frame_display)
    cv2.imshow('Mascara de Segmentacion', mascara)

    if cv2.waitKey(30) & 0xFF == 27: # Tecla ESC para salir
        break

cap.release()
cv2.destroyAllWindows()

# 7. EXPORTACIÓN DE RESULTADOS Y GRÁFICAS CINEMÁTICAS 
if len(posiciones_x) >= 3:
    print("\n[INFO] Generando gráficas cinemáticas...")
    
    pos_m, vel, acel = calcular_cinematica(posiciones_x, fps, FACTOR_ESCALA)
    
    tiempo = np.arange(len(pos_m)) / fps
    tiempo_vel = tiempo[:-1]
    tiempo_acel = tiempo_vel[:-1]
    
    vel_kmh = vel * 3.6

    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle('Análisis Cinemático del Vehículo', fontsize=16)
    
    axs[0].plot(tiempo, pos_m, 'b-', linewidth=2)
    axs[0].set_title('Posición vs Tiempo')
    axs[0].set_ylabel('Posición (m)')
    axs[0].grid(True)
    
    axs[1].plot(tiempo_vel, vel_kmh, 'g-', linewidth=2)
    axs[1].set_title('Velocidad vs Tiempo')
    axs[1].set_ylabel('Velocidad (km/h)')
    axs[1].grid(True)
    
    axs[2].plot(tiempo_acel, acel, 'r-', linewidth=2)
    axs[2].set_title('Aceleración vs Tiempo')
    axs[2].set_xlabel('Tiempo (s)')
    axs[2].set_ylabel('Aceleración (m/s²)')
    axs[2].grid(True)
    
    plt.tight_layout()
    ruta_guardado = os.path.join(SCRIPT_DIR, '..', 'results', 'graficas_cinematicas.png')
    plt.savefig(ruta_guardado)
    print(f"[INFO] Gráficas exportadas exitosamente en: {ruta_guardado}")
    plt.show()