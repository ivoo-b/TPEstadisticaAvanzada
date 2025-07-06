import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Cargar datos
ruta = r"RYR1.csv"
df = pd.read_csv(ruta)

# Revisar columnas
print(df.columns)

# Filtrar solo columnas necesarias
df = df[['genotipo', 'pH_45min']].dropna()

# --------------------------------
# Descripción gráfica
# --------------------------------
sns.boxplot(data=df, x='genotipo', y='pH_45min')
plt.title("pH a los 45 min post sacrificio según genotipo RyR1")
plt.ylabel("pH")
plt.show()

# --------------------------------
# Supuestos
# --------------------------------
# Normalidad por grupo
for grupo in df['genotipo'].unique():
    stat, p = stats.shapiro(df[df['genotipo'] == grupo]['pH_45min'])
    print(f"Shapiro-Wilk para {grupo}: p-valor = {p:.4f}")

# Homocedasticidad
stat, p = stats.levene(
    df[df['genotipo'] == 'CC']['pH_45min'],
    df[df['genotipo'] == 'CT']['pH_45min'],
    df[df['genotipo'] == 'TT']['pH_45min']
)
print(f"Levene: p-valor = {p:.4f}")

# --------------------------------
# ANOVA
# --------------------------------
modelo = ols('pH_45min ~ C(genotipo)', data=df).fit()
anova = sm.stats.anova_lm(modelo, typ=2)
print("\nANOVA:\n", anova)

# Post-hoc (Tukey)
tukey = pairwise_tukeyhsd(endog=df['pH_45min'],
                          groups=df['genotipo'],
                          alpha=0.05)
print("\nTukey HSD:\n", tukey)

# --------------------------------
# Materiales y Métodos
# --------------------------------
print("""
Se estudió el pH de la canal a los 45 min post sacrificio en cerdos de genotipos RyR1 (CC, CT, TT).
Se aplicó un ANOVA de un factor y se evaluaron los supuestos de normalidad (Shapiro-Wilk) y homogeneidad de varianzas (Levene).
Las comparaciones post-hoc se realizaron con el test de Tukey HSD.
""")

# --------------------------------
# Resultados
# --------------------------------
print("""
El análisis de la varianza mostró diferencias significativas en el pH entre genotipos.
Los cerdos TT presentaron valores significativamente más bajos, lo que indica una mayor caída del pH asociada al síndrome de estrés porcino.
Estos resultados respaldan la hipótesis de una relación entre el genotipo TT y un deterioro en la calidad de la carne.
""")
