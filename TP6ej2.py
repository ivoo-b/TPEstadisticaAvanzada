# problema2_tp6.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# cargar datos
ruta = r'Ru.id.csv'
datos = pd.read_csv(ruta)

# ver estructura
print(datos.head())

# gráfico de dispersión
plt.figure(figsize=(8,5))
sns.scatterplot(data=datos, x='RU', y='ID')
plt.title('Daño en ADN vs concentración de RU')
plt.xlabel('Concentración RU (µg/huevo)')
plt.ylabel('Índice de daño')
plt.grid(True)
plt.show()

# modelo lineal (RU como variable numérica)
modelo_num = smf.ols('ID ~ RU', data=datos).fit()
print(modelo_num.summary())

# residuos y supuestos
residuos = modelo_num.resid
ajustados = modelo_num.fittedvalues

# residuos vs ajustados
plt.figure(figsize=(8,5))
sns.residplot(x=ajustados, y=residuos, lowess=True, line_kws={'color':'red'})
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('Valores ajustados')
plt.ylabel('Residuos')
plt.title('Residuos vs Valores ajustados')
plt.show()

# histograma residuos
sns.histplot(residuos, kde=True)
plt.title('Distribución de residuos')
plt.xlabel('Residuo')
plt.show()

# test de normalidad
stat, p = stats.shapiro(residuos)
print(f'Shapiro-Wilk: estadístico={stat:.4f}, p={p:.4f}')

# R²
print(f"R² del modelo lineal: {modelo_num.rsquared:.4f}")

# predicciones para 1500 y 2200
pred_df = pd.DataFrame({'RU': [1500, 2200]})
predicciones = modelo_num.predict(pred_df)
for ru, y in zip(pred_df['RU'], predicciones):
    print(f"Para {ru} µg/huevo, se predice ID ≈ {y:.2f}")

# ahora RU como FACTOR
datos['RU_cat'] = datos['RU'].astype(str)
modelo_cat = smf.ols('ID ~ RU_cat', data=datos).fit()
anova_cat = sm.stats.anova_lm(modelo_cat, typ=2)
print("\nANOVA considerando RU como factor:")
print(anova_cat)

# boxplot por grupo
plt.figure(figsize=(10,5))
sns.boxplot(data=datos, x='RU_cat', y='ID')
plt.xticks(rotation=45)
plt.title('Índice de daño según dosis de RU (categoría)')
plt.xlabel('Dosis RU (µg/huevo)')
plt.ylabel('Índice de daño')
plt.tight_layout()
plt.show()

# comparar enfoques
print(f"\nR² modelo numérico: {modelo_num.rsquared:.4f}")
print(f"R² modelo categórico: {modelo_cat.rsquared:.4f}")
