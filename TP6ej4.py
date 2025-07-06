# problema4_tp6.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import shapiro

# cargar datos
ruta = r'aedes.csv'
datos = pd.read_csv(ruta)

# vista previa
print(datos.head())

# gráfico de dispersión
plt.figure(figsize=(8, 5))
sns.scatterplot(x='temperatura', y='tiempo_promedio', data=datos)
plt.title('Tiempo de desarrollo promedio vs Temperatura')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Tiempo promedio (hs)')
plt.grid(True)
plt.show()

# regresión lineal
modelo = smf.ols('tiempo_promedio ~ temperatura', data=datos).fit()
print(modelo.summary())

# residuos vs ajustados
residuos = modelo.resid
ajustados = modelo.fittedvalues

plt.figure(figsize=(8, 5))
sns.residplot(x=ajustados, y=residuos, lowess=True, line_kws={'color': 'red'})
plt.axhline(0, linestyle='--', color='black')
plt.title('Residuos vs Ajustados')
plt.xlabel('Valores ajustados')
plt.ylabel('Residuos')
plt.show()

# histograma de residuos
sns.histplot(residuos, kde=True)
plt.title('Distribución de residuos')
plt.xlabel('Residuos')
plt.show()

# test de normalidad de residuos
stat, pval = shapiro(residuos)
print(f'Test de Shapiro-Wilk: estadístico={stat:.4f}, p-valor={pval:.4f}')

# coeficientes
b0 = modelo.params['Intercept']
b1 = modelo.params['temperatura']
print(f"\nEcuación estimada: Y = {b0:.4f} + {b1:.4f} * X")

# interpretación de la pendiente
print(f"La pendiente (β1) indica que por cada grado más de temperatura, el tiempo promedio baja en {abs(b1):.4f} horas.")

# R²
print(f"R²: {modelo.rsquared:.4f} → indica que el modelo explica el {modelo.rsquared * 100:.1f}% de la variabilidad.")

# intervalo de confianza del 95% para la pendiente
confint_pendiente = modelo.conf_int().loc['temperatura']
print(f"IC del 95% para la pendiente: {confint_pendiente[0]:.4f} a {confint_pendiente[1]:.4f} hs/°C")
