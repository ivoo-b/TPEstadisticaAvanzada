# problema3_tp6.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import shapiro

# cargar datos
ruta = r'ratones_etanol.csv'
datos = pd.read_csv(ruta)

# vista previa
print(datos.head())

# gráfico de dispersión
plt.figure(figsize=(8,5))
sns.scatterplot(data=datos, x='dosis', y='volumen_cerebral')
plt.title('Volumen cerebral vs Dosis de Etanol')
plt.xlabel('Dosis de Etanol (g/kg)')
plt.ylabel('Volumen cerebral (cm³)')
plt.grid(True)
plt.show()

# regresión lineal
modelo = smf.ols('volumen_cerebral ~ dosis', data=datos).fit()
print(modelo.summary())

# residuos
residuos = modelo.resid
ajustados = modelo.fittedvalues

# supuestos: residuos vs ajustados
plt.figure(figsize=(8,5))
sns.residplot(x=ajustados, y=residuos, lowess=True, line_kws={'color':'red'})
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('Valores ajustados')
plt.ylabel('Residuos')
plt.title('Residuos vs Ajustados')
plt.show()

# histograma de residuos
sns.histplot(residuos, kde=True)
plt.title('Distribución de residuos')
plt.xlabel('Residuos')
plt.show()

# test de normalidad de residuos
stat, p = shapiro(residuos)
print(f'Test de Shapiro-Wilk: estadístico={stat:.4f}, p={p:.4f}')

# R²
print(f'R² del modelo: {modelo.rsquared:.4f}')

# predicciones con IC del 95%
nuevas_dosis = pd.DataFrame({'dosis': [1.5, 2.0, 5.0]})
pred = modelo.get_prediction(nuevas_dosis)
pred_summary = pred.summary_frame(alpha=0.05)
print("\nPredicciones con IC del 95%:")
print(pred_summary[['mean', 'mean_ci_lower', 'mean_ci_upper']])

# intervalo de confianza para la pendiente
confint_pendiente = modelo.conf_int(alpha=0.05).loc['dosis']
print(f"\nIC del 95% para la pendiente: {confint_pendiente[0]:.4f} a {confint_pendiente[1]:.4f} cm³ por g/kg")
