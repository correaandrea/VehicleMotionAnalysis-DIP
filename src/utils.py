import numpy as np

def calcular_centroide(mascara):
    """
    Calcula el centroide (x, y) de la mancha blanca más grande en la máscara.
    Punto 2 - Requisito 20.
    """
    import cv2
    # Encontrar contornos en la máscara [cite: 19]
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contornos:
        # Tomar el contorno con mayor área para evitar ruido [cite: 36]
        cnt = max(contornos, key=cv2.contourArea)
        # Calcular momentos [cite: 20]
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy), cnt
    return None, None

def calcular_cinematica(posiciones_px, fps, factor_escala):
    """
    Calcula velocidad y aceleración usando diferencias finitas.
    Punto 3 - Requisitos 23 y 24.
    """
    dt = 1.0 / fps # Tiempo entre fotogramas [cite: 14]
    
    # Convertir posiciones de píxeles a metros [cite: 22, 37]
    posiciones_m = np.array(posiciones_px) * factor_escala
    
    # Velocidad (v = dx/dt) usando diferencias finitas hacia adelante [cite: 23]
    velocidades = np.diff(posiciones_m) / dt
    
    # Aceleración (a = dv/dt) como segunda derivada [cite: 24]
    aceleraciones = np.diff(velocidades) / dt
    
    return posiciones_m, velocidades, aceleraciones