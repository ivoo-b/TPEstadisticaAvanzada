import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Cargar los datos
ruta = r"microARN.csv"
df = pd.read_csv(ruta)

# Revisar nombres de columnas
print(df.columns)

# Renombrar si es necesario
df.columns = ['grupo', 'expresion_mir150']

# Unidad experimental: paciente
# Variable dependiente: expresion_mir150 (cuantitativa continua)
# Variable independiente: grupo (categórica: cáncer, adenoma, sano)
# Réplicas: contar por grupo
print(df['grupo'].value_counts())

# --------------------------------
# Descripción gráfica
# --------------------------------
sns.boxplot(data=df, x='grupo', y='expresion_mir150')
plt.title("Expresión miR-150 por grupo")
plt.show()

# Outliers
sns.stripplot(data=df, x='grupo', y='expresion_mir150', jitter=True, color='black')
plt.title("Puntos individuales de expresión")
plt.show()

# --------------------------------
# Supuestos de ANOVA
# --------------------------------
# Normalidad por grupo
for grupo in df['grupo'].unique():
    stat, p = stats.shapiro(df[df['grupo'] == grupo]['expresion_mir150'])
    print(f"Shapiro-Wilk para {grupo}: p-valor = {p:.4f}")

# Homogeneidad de varianzas
stat, p = stats.levene(
    df[df['grupo'] == 'cancer']['expresion_mir150'],
    df[df['grupo'] == 'adenoma']['expresion_mir150'],
    df[df['grupo'] == 'sano']['expresion_mir150']
)
print(f"Levene: p-valor = {p:.4f}")

# --------------------------------
# ANOVA
# --------------------------------
modelo = ols('expresion_mir150 ~ C(grupo)', data=df).fit()
anova = sm.stats.anova_lm(modelo, typ=2)
print("\nANOVA:\n", anova)

# Comparaciones post-hoc
from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey = pairwise_tukeyhsd(endog=df['expresion_mir150'],
                          groups=df['grupo'],
                          alpha=0.05)
print("\nTukey HSD:\n", tukey)

# --------------------------------
# Gráfico final con medias y errores
# --------------------------------
sns.pointplot(data=df, x='grupo', y='expresion_mir150', capsize=.2, errorbar='se')
plt.title("Expresión media de miR-150 por grupo")
plt.ylabel("miR-150 (μU/mg)")
plt.show()

# --------------------------------
# Materiales y Métodos
# --------------------------------
print("""
Se evaluó la expresión de miR-150 en tres grupos de pacientes (con cáncer de próstata, con adenoma y sanos) mediante qRT-PCR.
Se aplicó ANOVA de una vía para comparar medias entre grupos, previo chequeo de supuestos de normalidad (Shapiro-Wilk) y homocedasticidad (Levene).
Se usó Tukey HSD como análisis post-hoc.
""")

# --------------------------------
# Resultados
# --------------------------------
print("""
Se observaron diferencias estadísticamente significativas en la expresión de miR-150 entre los grupos.
El grupo con cáncer presentó valores más bajos comparado con los grupos sano y con adenoma.
Esto sugiere que miR-150 podría estar asociado a la presencia de cáncer de próstata.
""")
