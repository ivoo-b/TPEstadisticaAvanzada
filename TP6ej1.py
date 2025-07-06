# problema1_tp6.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

# cargar los datos
ruta = r'curva_calib_AF.csv'
datos = pd.read_csv(ruta)

# convertir dosis de µg a mg
datos['dosis_mg'] = datos['Dosis'] / 1000

# gráfico de dispersión
plt.figure(figsize=(8,5))
sns.scatterplot(data=datos, x='dosis_mg', y='Respuesta')
plt.title('Curva de calibración: dosis (mg) vs. respuesta')
plt.xlabel('Dosis (mg)')
plt.ylabel('Respuesta')
plt.grid(True)
plt.show()

# regresión lineal
X = sm.add_constant(datos['dosis_mg'])  # agregar término independiente β0
modelo = sm.OLS(datos['Respuesta'], X).fit()
print(modelo.summary())

# interpretar coeficientes
b0, b1 = modelo.params
print(f"Ecuación estimada: E[Y] = {b0:.4f} + {b1:.4f} * dosis_mg")

# verificar supuestos
residuos = modelo.resid
ajustados = modelo.fittedvalues

# gráfico de residuos vs ajustados
plt.figure(figsize=(8,5))
sns.residplot(x=ajustados, y=residuos, lowess=True, line_kws={'color': 'red'})
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('Valores ajustados')
plt.ylabel('Residuos')
plt.title('Residuos vs. Valores ajustados')
plt.show()

# histograma y test de normalidad de residuos
plt.figure(figsize=(6,4))
sns.histplot(residuos, kde=True)
plt.title('Distribución de residuos')
plt.xlabel('Residuo')
plt.show()

# test de Shapiro-Wilk
stat, p = stats.shapiro(residuos)
print(f'Shapiro-Wilk: estadístico={stat:.4f}, p={p:.4f}')

# coeficiente de determinación R²
r2 = modelo.rsquared
print(f"R² = {r2:.4f} → el modelo explica el {r2*100:.2f}% de la variabilidad")

# predicciones
nueva_dosis = [15, 350]  # mg
nuevas = pd.DataFrame({'const': 1, 'dosis_mg': nueva_dosis})
predicciones = modelo.predict(nuevas)
for d, y in zip(nueva_dosis, predicciones):
    print(f'Para {d} mg, la lectura esperada es {y:.2f}')

# estimar concentración de una muestra incógnita con lectura 265
lectura_dada = 265
estimada = (lectura_dada - b0) / b1
print(f'Para una lectura de {lectura_dada}, la concentración estimada es {estimada:.2f} mg')
