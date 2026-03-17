import cv2
import numpy as np

def calcular_centroide(mascara):

    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contornos:
        return None, None

    contorno_valido = None
    area_max = 0

    for cnt in contornos:

        area = cv2.contourArea(cnt)

        if area > 1500:  # filtra ruido

            if area > area_max:
                area_max = area
                contorno_valido = cnt

    if contorno_valido is None:
        return None, None

    M = cv2.moments(contorno_valido)

    if M["m00"] == 0:
        return None, None

    cx = int(M["m10"]/M["m00"])
    cy = int(M["m01"]/M["m00"])

    return (cx,cy), contorno_valido