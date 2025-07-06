# problema6_tp2.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# parámetros
media_poblacional = 100
varianza = 100
desvio = np.sqrt(varianza)
n = 10
m = 100  # cantidad de muestras
z = norm.ppf(1 - 0.05 / 2)

# simular
ICs = []
aciertos = 0

for _ in range(m):
    muestra = np.random.normal(loc=media_poblacional, scale=desvio, size=n)
    media_muestra = np.mean(muestra)
    error = desvio / np.sqrt(n)
    LI = media_muestra - z * error
    LS = media_muestra + z * error
    ICs.append((LI, LS))
    if LI <= media_poblacional <= LS:
        aciertos += 1

# graficar
plt.figure(figsize=(10, 6))
for i, (LI, LS) in enumerate(ICs):
    color = 'blue' if LI <= media_poblacional <= LS else 'red'
    plt.plot([i, i], [LI, LS], color=color)
plt.axhline(media_poblacional, color='black', linestyle='--')
plt.title(f"Simulación de IC para la media (Cobertura: {aciertos}/{m})")
plt.xlabel("Muestra")
plt.ylabel("Intervalo de confianza")
plt.show()
