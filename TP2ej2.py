# problema1_tp2.py
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# ----------- parámetros de simulación ----------- #
tamano_poblacion = 100000
media_poblacion = 50
desvio_poblacion = 20

tamano_muestra = 5
cantidad_muestras = 10000

# ----------- generar población NO normal ----------- #
# usamos una distribución exponencial para simular asimetría
poblacion = np.random.exponential(scale=media_poblacion, size=tamano_poblacion)

# ----------- simular distribución muestral de la media ----------- #
medias_muestrales = []

for _ in range(cantidad_muestras):
    muestra = np.random.choice(poblacion, size=tamano_muestra)
    medias_muestrales.append(np.mean(muestra))

# ----------- graficar ----------- #
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# histograma de la población
axs[0].hist(poblacion, bins=50, color='skyblue', edgecolor='black', density=True)
axs[0].set_title('Distribución de la población (Exponencial)')
axs[0].axvline(np.mean(poblacion), color='blue', linestyle='--', label=f'Media = {np.mean(poblacion):.2f}')
axs[0].legend()

# histograma de una muestra aleatoria
muestra_ejemplo = np.random.choice(poblacion, size=tamano_muestra)
axs[1].hist(muestra_ejemplo, bins=10, color='lightgreen', edgecolor='black')
axs[1].set_title(f'Muestra aleatoria (n={tamano_muestra})')
axs[1].axvline(np.mean(muestra_ejemplo), color='red', linestyle='--', label=f'Media = {np.mean(muestra_ejemplo):.2f}')
axs[1].legend()

# distribución muestral de la media
axs[2].hist(medias_muestrales, bins=50, color='salmon', edgecolor='black', density=True)
axs[2].set_title(f'Distribución muestral de la media\n({cantidad_muestras} muestras de tamaño {tamano_muestra})')
media_muestral = np.mean(medias_muestrales)
std_muestral = np.std(medias_muestrales)
axs[2].axvline(media_muestral, color='blue', linestyle='--', label=f'Media = {media_muestral:.2f}')
x = np.linspace(min(medias_muestrales), max(medias_muestrales), 500)
axs[2].plot(x, stats.norm.pdf(x, media_muestral, std_muestral), color='black', linestyle='-', label='Normal Teórica')
axs[2].legend()

plt.tight_layout()
plt.show()

# ----------- interpretación sugerida ----------- #
print(f"Media de la población: {np.mean(poblacion):.2f}")
print(f"Media muestral promedio (n={tamano_muestra}): {media_muestral:.2f}")
print(f"Desvío estándar de las medias muestrales: {std_muestral:.2f}")
