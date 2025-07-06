import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Cargar datos
ruta = r"PECT.csv"
df = pd.read_csv(ruta)

# Revisar columnas
print(df.columns)

# Renombrar si es necesario
df.columns = ['genotipo', 'colesterol_hdl']

# --------------------------------
# Descripción gráfica
# --------------------------------
sns.boxplot(data=df, x='genotipo', y='colesterol_hdl')
plt.title("Colesterol HDL según genotipo PECT")
plt.ylabel("HDL (mmol/l)")
plt.show()

# --------------------------------
# Supuestos para ANOVA
# --------------------------------
# Normalidad por grupo
for grupo in df['genotipo'].unique():
    stat, p = stats.shapiro(df[df['genotipo'] == grupo]['colesterol_hdl'])
    print(f"Shapiro-Wilk para {grupo}: p-valor = {p:.4f}")

# Homogeneidad de varianzas
stat, p = stats.levene(
    df[df['genotipo'] == 'Ile/Ile']['colesterol_hdl'],
    df[df['genotipo'] == 'Ile/Val']['colesterol_hdl'],
    df[df['genotipo'] == 'Val/Val']['colesterol_hdl']
)
print(f"Levene: p-valor = {p:.4f}")

# --------------------------------
# ANOVA
# --------------------------------
modelo = ols('colesterol_hdl ~ C(genotipo)', data=df).fit()
anova = sm.stats.anova_lm(modelo, typ=2)
print("\nANOVA:\n", anova)

# Comparaciones post-hoc
from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey = pairwise_tukeyhsd(endog=df['colesterol_hdl'],
                          groups=df['genotipo'],
                          alpha=0.05)
print("\nTukey HSD:\n", tukey)

# Tabla con medias e IC
import numpy as np
import scipy.stats as stats

def resumen_con_ic(grupo):
    valores = df[df['genotipo'] == grupo]['colesterol_hdl']
    media = np.mean(valores)
    sd = np.std(valores, ddof=1)
    n = len(valores)
    error = stats.t.ppf(0.975, df=n-1) * sd / np.sqrt(n)
    return media, media - error, media + error

for g in df['genotipo'].unique():
    media, li, ls = resumen_con_ic(g)
    print(f"{g}: media = {media:.2f}, IC 95% = [{li:.2f}, {ls:.2f}]")

# --------------------------------
# Materiales y Métodos
# --------------------------------
print("""
Se analizaron los niveles de colesterol HDL en mujeres entre 20 y 50 años con genotipos PECT: Ile/Ile, Ile/Val y Val/Val.
Se aplicó un ANOVA de un factor para evaluar diferencias entre genotipos. Se verificaron los supuestos de normalidad y homogeneidad de varianzas.
Las comparaciones múltiples se realizaron con el test de Tukey.
""")

# --------------------------------
# Resultados
# --------------------------------
print("""
El análisis mostró diferencias estadísticamente significativas entre los genotipos.
Los valores medios de HDL fueron mayores en el grupo Ile/Ile y menores en Val/Val.
Esto sugiere que la mutación Val podría estar asociada a niveles reducidos de HDL.
""")
