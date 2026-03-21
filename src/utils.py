import cv2
import numpy as np

def calcular_centroide(mascara):
    """
    Calcula el centroide del objeto principal en una máscara binaria.
    
    Utiliza detección de contornos y momentos espaciales para encontrar
    el centro de masa del vehículo segmentado, ignorando artefactos de ruido.
    
    Args:
        mascara (numpy.ndarray): Imagen binaria (blanco/negro) obtenida de la segmentación.
        
    Returns:
        tuple: ((cx, cy), contorno_valido) donde (cx, cy) son las coordenadas del centroide
               y contorno_valido es el arreglo de puntos que delimita al vehículo.
               Retorna (None, None) si no se detecta un objeto válido.
    """
    # Extraer los contornos externos de la máscara binaria
    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,       # Recuperar solo los contornos más externos (ignorar huecos internos)
        cv2.CHAIN_APPROX_SIMPLE  # Comprimir segmentos para ahorrar memoria
    )

    if not contornos:
        return None, None

    contorno_valido = None
    area_max = 0

    # Evaluar todos los contornos encontrados para aislar el vehículo
    for cnt in contornos:
        area = cv2.contourArea(cnt)

        # Umbral de área (300 px) para descartar ruido residual del asfalto o reflejos
        if area > 300:  
            # Quedarse estrictamente con el contorno de mayor tamaño (el carro)
            if area > area_max:
                area_max = area
                contorno_valido = cnt

    if contorno_valido is None:
        return None, None

    # Calcular los momentos espaciales de la imagen para el contorno seleccionado
    M = cv2.moments(contorno_valido)

    # Evitar división por cero en caso de un contorno anómalo sin área efectiva
    if M["m00"] == 0:
        return None, None

    # Fórmulas matemáticas para el centro de masa (centroide): cx = M10/M00, cy = M01/M00
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    return (cx, cy), contorno_valido


def calcular_cinematica(posiciones_x, fps, factor_escala):
    """
    Realiza el análisis cinemático 1D (eje X) a partir del historial de posiciones.
    
    Aplica derivación numérica para estimar la velocidad y la aceleración instantánea 
    del vehículo a lo largo del tiempo.
    
    Args:
        posiciones_x (list): Lista de coordenadas X del centroide en píxeles.
        fps (float): Tasa de fotogramas por segundo del video original.
        factor_escala (float): Relación de conversión de píxeles a metros (m/px).
        
    Returns:
        tuple: (posiciones_m, velocidades, aceleraciones) como arreglos de NumPy.
    """
    # Se requieren al menos 3 puntos en el tiempo para poder derivar hasta la aceleración
    if len(posiciones_x) < 3:
        return np.array([]), np.array([]), np.array([])
        
    # Diferencial de tiempo (dt) entre cada fotograma capturado
    dt = 1.0 / fps 
    
    # Transformación del espacio de imagen (píxeles) al espacio físico (metros)
    posiciones_m = np.array(posiciones_x) * factor_escala
    
    # Primera derivada numérica (v = dx/dt): Velocidad en m/s
    velocidades = np.diff(posiciones_m) / dt
    
    # Segunda derivada numérica (a = dv/dt): Aceleración en m/s^2
    aceleraciones = np.diff(velocidades) / dt
    
    return posiciones_m, velocidades, aceleraciones