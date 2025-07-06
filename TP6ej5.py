# problema5_tp6.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, linregress

# datos ingresados manualmente
lignina = [
    8.01, 7.85, 12.56, 13.33, 8.22, 9.53, 9.63, 12.48, 9.5, 15.4,
    9.36, 9.35, 10.74, 6.62, 11.9, 12.3, 7.88, 9.84, 10.69, 10.69
]

volumen = [
    12.46, 11.91, 9.71, 8.15, 10.99, 10.27, 9.84, 9.19, 10.51, 8.49,
    10.96, 10.69, 10.03, 13.0, 8.85, 9.78, 11.55, 9.84, 10.69, 9.78
]

# armar dataframe
df = pd.DataFrame({'lignina': lignina, 'volumen': volumen})

# gráfico de dispersión
plt.figure(figsize=(8, 5))
sns.scatterplot(x='lignina', y='volumen', data=df)
plt.title('Relación entre contenido de lignina y volumen de líquido perdido')
plt.xlabel('Contenido de lignina (mg/g)')
plt.ylabel('Volumen perdido (ml/100g)')
plt.grid(True)
plt.show()

# coeficiente de correlación
r, p = pearsonr(df['lignina'], df['volumen'])
print(f"Coeficiente de correlación de Pearson: r = {r:.4f}")
print(f"P-valor: {p:.4f}")
if p < 0.05:
    print("→ La asociación es estadísticamente significativa.")
else:
    print("→ No hay evidencia suficiente de asociación.")

# regresión lineal
slope, intercept, r_value, p_value, std_err = linregress(df['lignina'], df['volumen'])
print(f"\nEcuación de regresión: Y = {intercept:.4f} + {slope:.4f} * X")
print(f"R²: {r_value**2:.4f} → el modelo explica el {r_value**2 * 100:.2f}% de la variabilidad")

# graficar con línea de regresión
plt.figure(figsize=(8, 5))
sns.regplot(x='lignina', y='volumen', data=df, ci=95, line_kws={'color': 'red'})
plt.title('Regresión lineal: volumen perdido vs lignina')
plt.xlabel('Contenido de lignina (mg/g)')
plt.ylabel('Volumen perdido (ml/100g)')
plt.grid(True)
plt.show()
