import cv2
import numpy as np
import os

# --- 1. CONFIGURACIÓN DE RUTAS ---
# Usamos rutas relativas para que funcione en las PCs de los 3 integrantes
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Asegúrate de que el video esté en la carpeta /data con este nombre
VIDEO_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'VideoPDI.mp4')

# --- 2. VARIABLES GLOBALES ---
pausar_video = False
mostrar_pixel = False
x_pixel, y_pixel = 0, 0

# --- 3. FUNCIONES DE INTERACCIÓN (CALLBACKS) ---
def mouse_callback(event, x, y, flags, param):
    global pausar_video, mostrar_pixel, x_pixel, y_pixel
    # Click Izquierdo: Captura posición y color (útil para el Punto 2 y 3) 
    if event == cv2.EVENT_LBUTTONDOWN:
        x_pixel, y_pixel = x, y
        mostrar_pixel = True
    # Click Derecho: Pausa el video para analizar mejor un frame
    if event == cv2.EVENT_RBUTTONDOWN:
        pausar_video = not pausar_video

def nothing(x):
    pass

# --- 4. INICIALIZACIÓN DE VIDEO ---
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error crítico: No se encontró el video en {VIDEO_PATH}")
    print("Verifica que el nombre sea VideoPDI.mp4 y esté dentro de la carpeta /data")
    exit()

# --- 5. CONFIGURACIÓN DE PROPORCIÓN (FIX ASPECT RATIO) ---
# Definimos un ancho fijo y calculamos el alto para evitar que el video se estire
ancho_visualizacion = 800 
ancho_original = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
alto_original = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

aspect_ratio = alto_original / ancho_original
alto_visualizacion = int(ancho_visualizacion * aspect_ratio)

# --- 6. VENTANAS Y CONTROLES (TRACKBARS) ---
cv2.namedWindow('Video')
cv2.setMouseCallback('Video', mouse_callback)

# Ventana para calibrar la segmentación HSV 
cv2.namedWindow("Trackbars")
cv2.createTrackbar("H Min", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("H Max", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("S Min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("S Max", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("V Min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("V Max", "Trackbars", 255, 255, nothing)

print(f"Procesando video a {ancho_visualizacion}x{alto_visualizacion} px.")

# --- 7. BUCLE PRINCIPAL ---
while True:
    if not pausar_video:
        ret, frame = cap.read()
        if not ret:
            # Si el video termina, vuelve a empezar (loop)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # Redimensionar manteniendo la proporción para no deformar el vehículo
        frame_resized = cv2.resize(frame, (ancho_visualizacion, alto_visualizacion))
        
    # Crear copia para no rayar el frame original con texto
    frame_display = frame_resized.copy()

    # Mostrar información del píxel si se hizo click [cite: 22]
    if mostrar_pixel:
        # Obtener color en formato BGR del frame redimensionado
        color_bgr = frame_resized[y_pixel, x_pixel]
        # Convertir ese frame a HSV solo para leer el valor del píxel seleccionado [cite: 16]
        temp_hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
        color_hsv = temp_hsv[y_pixel, x_pixel]
        
        cv2.putText(frame_display, f'Pos:({x_pixel},{y_pixel}) HSV:{color_hsv}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # SEGMENTACIÓN: Convertir a espacio HSV [cite: 16]
    hsv_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)

    # Leer valores de los Trackbars para el umbralamiento (Thresholding) [cite: 17]
    h_min = cv2.getTrackbarPos("H Min", "Trackbars")
    h_max = cv2.getTrackbarPos("H Max", "Trackbars")
    s_min = cv2.getTrackbarPos("S Min", "Trackbars")
    s_max = cv2.getTrackbarPos("S Max", "Trackbars")
    v_min = cv2.getTrackbarPos("V Min", "Trackbars")
    v_max = cv2.getTrackbarPos("V Max", "Trackbars")

    # Definir rangos y crear la máscara binaria [cite: 17]
    rango_bajo = np.array([h_min, s_min, v_min])
    rango_alto = np.array([h_max, s_max, v_max])
    mascara = cv2.inRange(hsv_frame, rango_bajo, rango_alto)

    # MOSTRAR VENTANAS
    cv2.imshow('Video', frame_display)
    cv2.imshow('Mascara (Segmentacion)', mascara)

    # Salir con la tecla ESC
    if cv2.waitKey(30) & 0xFF == 27:
        break

# --- 8. LIMPIEZA ---
cap.release()
cv2.destroyAllWindows()