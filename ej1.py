# problema1.py
# Trabajo Práctico 1 - Estadística descriptiva
# Problema 1: análisis de concentración de fosfolípidos en aceite de girasol

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ruta del archivo CSV
ruta_csv = r"girasol.csv"

# Cargar los datos
datos = pd.read_csv(ruta_csv)
print("Primeras filas del archivo:\n", datos.head())

# Asegurarse de que la columna se llama correctamente
columna = datos.columns[0]  # debería ser algo como 'fosfolipidos'
valores = datos[columna]

# 1. Identificación:
print("\nVariable:", columna, "(cuantitativa continua)")
print("Unidad de observación: alícuotas de aceite crudo de girasol de 10 ml")
print("Tamaño de la muestra:", len(valores))
print("Población: todas las alícuotas posibles del lote analizado")

# 2. Gráfico y análisis visual
plt.hist(valores, bins=10, edgecolor='black')
plt.axvline(np.mean(valores), color='red', linestyle='dashed', linewidth=1, label='Media')
plt.axvline(np.median(valores), color='green', linestyle='dashed', linewidth=1, label='Mediana')
plt.title("Histograma de concentración de fosfolípidos")
plt.xlabel("g/100g")
plt.ylabel("Frecuencia")
plt.legend()
plt.grid(True)
plt.show()

# Boxplot para detectar outliers
plt.boxplot(valores, vert=False)
plt.title("Boxplot de concentración de fosfolípidos")
plt.xlabel("g/100g")
plt.grid(True)
plt.show()

# 3. Medidas estadísticas
media = np.mean(valores)
desvio = np.std(valores, ddof=1)  # muestral
print(f"\nMedia: {media:.4f} g/100g")
print(f"Desvío estándar: {desvio:.4f} g/100g")

# 4. Homogeneidad (coeficiente de variación)
cv = desvio / media
print(f"Coeficiente de variación: {cv:.2%} → {'homogénea' if cv < 0.2 else 'heterogénea'}")

# 5. Mediana
mediana = np.median(valores)
print(f"Mediana: {mediana:.4f} g/100g")

# 6. Porcentaje con fosfolípidos > 0.40 g/100g
superiores_040 = (valores > 0.40).sum()
porcentaje = superiores_040 / len(valores) * 100
print(f"\nCantidad de muestras con fosfolípidos > 0.40 g/100g: {superiores_040} ({porcentaje:.1f}%)")

# 7. Percentil 10
percentil_10 = np.percentile(valores, 10)
print(f"Percentil 10: {percentil_10:.4f} g/100g → el 10% de las muestras tienen menos de ese valor")
