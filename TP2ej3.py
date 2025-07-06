# problema3_tp2.py
import numpy as np
from scipy.stats import t

# datos de residuos (ppm)
datos = [
    0.58, 0.95, 0.46, 0.84, 0.59, 0.92, 0.52, 0.92, 0.52, 0.55,
    0.40, 0.51, 0.52, 0.52, 0.60, 0.70, 0.35, 0.40, 0.50, 0.41,
    0.53, 0.51, 0.66, 0.60, 0.45, 0.77, 0.39, 0.50, 0.66, 0.85
]

datos = np.array(datos)
n = len(datos)
media = np.mean(datos)
desvio = np.std(datos, ddof=1)
nivel_confianza = 0.99

# intervalo de confianza para la media
alpha = 1 - nivel_confianza
t_critico = t.ppf(1 - alpha / 2, df=n - 1)
error_estandar = desvio / np.sqrt(n)
LI = media - t_critico * error_estandar
LS = media + t_critico * error_estandar

# resultado
print(f"Media muestral: {media:.4f}")
print(f"Desvío estándar: {desvio:.4f}")
print(f"IC del 99%: ({LI:.4f}, {LS:.4f})")

# evaluación del lote
if LI > 0.50:
    print("→ El lote NO es apto para consumo (todas las posibles medias superan el límite).")
elif LS < 0.50:
    print("→ El lote es apto (todas las medias posibles están por debajo del límite).")
else:
    print("→ El lote podría no ser apto (el IC incluye valores mayores a 0.50).")
