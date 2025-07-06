# problema5_tp2.py
import numpy as np
from scipy.stats import norm

z = norm.ppf(1 - 0.05 / 2)  # z para 95%

# estimación de proporción: prematuros
p = 0.5  # caso más conservador (p desconocida)
margen_error_proporcion = 0.05
n_proporcion = (z ** 2 * p * (1 - p)) / (margen_error_proporcion ** 2)
n_proporcion = int(np.ceil(n_proporcion))

# estimación de media: peso al nacer
desvio_estimado = 500  # en gramos (se puede ajustar si hay info previa)
margen_error_media = 250
n_media = ((z * desvio_estimado) / margen_error_media) ** 2
n_media = int(np.ceil(n_media))

print("Parámetros:")
print("- Proporción de recién nacidos prematuros")
print("- Peso promedio de recién nacidos a término")

print("\nEstimadores:")
print("- Frecuencia relativa de prematuros")
print("- Media muestral de pesos")

print("\nTamaño muestral necesario:")
print(f"- Para estimar proporción ±5%: {n_proporcion} bebés")
print(f"- Para estimar peso con ±250g: {n_media} bebés")

print("\nNota: se necesitaría conocer la desviación estándar real del peso\npara afinar más el cálculo de n para la media.")
