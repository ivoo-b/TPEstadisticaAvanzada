import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Cargar los datos
ruta_csv = r"neuralgia.csv"
df = pd.read_csv(ruta_csv)

# Mostrar primeras filas
print(df.head())

# Verificar nombres de columnas
print(df.columns)

# Renombrar columnas si es necesario (por seguridad)
df.columns = ['tratamiento', 'horas_dolor', 'efectos_gastricos']

# Unidad experimental: paciente
# Variable independiente: tratamiento (categórica)
# Variable dependiente: horas_dolor (cuantitativa continua)
# Réplicas: 10 por tratamiento

# -------------------------------
# Análisis gráfico y exploratorio
# -------------------------------
sns.boxplot(data=df, x='tratamiento', y='horas_dolor')
plt.title("Horas sin dolor por tratamiento")
plt.show()

sns.histplot(df['horas_dolor'], kde=True)
plt.title("Distribución de horas sin dolor (todas las muestras)")
plt.xlabel("Horas sin dolor")
plt.show()

# -------------------------------
# Supuestos: normalidad y homogeneidad
# -------------------------------

# Normalidad por grupo
for grupo in df['tratamiento'].unique():
    stat, p = stats.shapiro(df[df['tratamiento'] == grupo]['horas_dolor'])
    print(f"Shapiro-Wilk para {grupo}: p-valor = {p:.4f}")

# Homogeneidad de varianzas
stat, p = stats.levene(
    df[df['tratamiento'] == 'Nueva droga']['horas_dolor'],
    df[df['tratamiento'] == 'Aspirina']['horas_dolor'],
    df[df['tratamiento'] == 'Placebo']['horas_dolor']
)
print(f"Levene test: p-valor = {p:.4f}")

# -------------------------------
# ANOVA
# -------------------------------
modelo = ols('horas_dolor ~ C(tratamiento)', data=df).fit()
anova = sm.stats.anova_lm(modelo, typ=2)
print("\nANOVA:\n", anova)

# -------------------------------
# Comparaciones post-hoc (Tukey)
# -------------------------------
from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey = pairwise_tukeyhsd(endog=df['horas_dolor'],
                          groups=df['tratamiento'],
                          alpha=0.05)
print("\nTukey HSD:\n", tukey)

# -------------------------------
# Frecuencia de efectos colaterales gástricos
# -------------------------------
sns.boxplot(data=df, x='tratamiento', y='efectos_gastricos')
plt.title("Efectos colaterales gástricos por tratamiento")
plt.show()

# ANOVA para efectos colaterales (tratamiento sobre escala ordinal)
modelo_efectos = ols('efectos_gastricos ~ C(tratamiento)', data=df).fit()
anova_efectos = sm.stats.anova_lm(modelo_efectos, typ=2)
print("\nANOVA - Efectos colaterales gástricos:\n", anova_efectos)

# -------------------------------
# Materiales y Métodos (resumen)
# -------------------------------
print("""
Se realizó un análisis de varianza (ANOVA) para comparar el efecto de tres tratamientos (Nueva droga, Aspirina y Placebo) sobre las horas libres de dolor en pacientes con neuralgia crónica.
Se verificaron los supuestos de normalidad (test de Shapiro-Wilk) y homogeneidad de varianzas (test de Levene).
Posteriormente se realizaron comparaciones múltiples con el test de Tukey.
También se aplicó ANOVA sobre los puntajes de efectos colaterales gástricos, evaluados en escala de 0 a 5.
""")
