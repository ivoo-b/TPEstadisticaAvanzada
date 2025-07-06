# problema7_tp6.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn
import numpy as np
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

# cargar datos
ruta = r'latinoam.csv'
df = pd.read_csv(ruta)

# descripción univariada
print("Resumen estadístico general:\n")
print(df.describe(include='all'))

# verificar datos faltantes
faltantes = df.isnull().sum()
print("\nCantidad de datos faltantes por columna:\n")
print(faltantes)

# matriz de correlación (solo variables numéricas)
datos_numericos = df.select_dtypes(include=[np.number])
matriz_corr = datos_numericos.corr()
print("\nMatriz de correlación:\n")
print(matriz_corr)

# gráfico de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Matriz de correlación - Latinoamérica")
plt.tight_layout()
plt.show()

# matriz de dispersión
sns.pairplot(datos_numericos)
plt.suptitle("Matriz de diagramas de dispersión - Latinoamérica", y=1.02)
plt.show()

# identificar las variables más asociadas
abs_corr = matriz_corr.abs()
np.fill_diagonal(abs_corr.values, 0)
max_var1, max_var2 = np.unravel_index(np.argmax(abs_corr.values), abs_corr.shape)
print(f"\nVariables más fuertemente asociadas: {abs_corr.columns[max_var1]} y {abs_corr.columns[max_var2]}")
print(f"Coeficiente de correlación: {matriz_corr.iloc[max_var1, max_var2]:.4f}")

# detección de outliers bivariados usando z-score (simplificado)
z_scores = np.abs(stats.zscore(datos_numericos, nan_policy='omit'))
outliers = (z_scores > 3).any(axis=1)
print(f"\nPaíses con valores atípicos bivariados (z-score > 3):")
print(df['Pais'][outliers].tolist())
