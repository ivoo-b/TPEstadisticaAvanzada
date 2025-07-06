# problema6_tp6.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_rel

# datos de proteína por técnica
quimico = [14.3, 4, 15.9, 14.3, 14.3, 13.6, 8, 13.3, 5.2, 10.8]
nirs = [11.4, 4.9, 15.3, 14.2, 10.3, 10, 8.4, 13.5, 5, 10.3]

# armar dataframe
df = pd.DataFrame({
    'muestra': list(range(1, 11)),
    'quimico': quimico,
    'nirs': nirs
})

# gráfico de dispersión
plt.figure(figsize=(6, 6))
sns.scatterplot(x='quimico', y='nirs', data=df)
plt.plot([min(quimico), max(quimico)], [min(quimico), max(quimico)], 'r--', label='y = x')
plt.title('Comparación entre métodos: Químico vs NIRS')
plt.xlabel('Método químico (mg%)')
plt.ylabel('NIRS (mg%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# correlación de Pearson
r, p_valor = pearsonr(df['quimico'], df['nirs'])
print(f"Coeficiente de correlación: r = {r:.4f}")
print(f"P-valor: {p_valor:.4f}")
if p_valor < 0.05:
    print("→ Correlación significativa entre ambos métodos.")
else:
    print("→ No hay correlación significativa.")

# prueba t pareada
t_stat, p_ttest = ttest_rel(df['quimico'], df['nirs'])
print(f"\nPrueba t pareada:")
print(f"t = {t_stat:.4f}, p = {p_ttest:.4f}")
if p_ttest < 0.05:
    print("→ Hay diferencia significativa entre los métodos.")
else:
    print("→ No hay diferencia significativa entre los métodos.")

# conclusión
if r > 0.9 and p_ttest >= 0.05:
    print("\nConclusión: NIRS puede ser utilizada como técnica alternativa al método químico.")
else:
    print("\nConclusión: Se requieren más pruebas para validar el uso rutinario de NIRS.")
