
# Problema 4 - Asociación entre genotipo RYR1 y sexo

import pandas as pd
import scipy.stats as stats

# Cargar el archivo CSV
ruta = r"RYR1.csv"
datos = pd.read_csv(ruta)

# Nos aseguramos de que las columnas necesarias existan
if 'sexo' not in datos.columns or 'genotipo' not in datos.columns:
    raise ValueError("Las columnas 'sexo' y 'genotipo' deben estar presentes en el CSV")

# Crear tabla de contingencia
tabla_contingencia = pd.crosstab(datos['sexo'], datos['genotipo'])

print("Tabla de contingencia sexo vs genotipo:")
print(tabla_contingencia)

# Prueba de chi-cuadrado de independencia
chi2, p, dof, expected = stats.chi2_contingency(tabla_contingencia)

print("\nResultados de la prueba de independencia:")
print(f"Estadístico chi-cuadrado: {chi2:.4f}")
print(f"Valor p: {p:.4f}")
print(f"Grados de libertad: {dof}")
print("\nFrecuencias esperadas bajo independencia:")
print(pd.DataFrame(expected, index=tabla_contingencia.index, columns=tabla_contingencia.columns))

if p < 0.05:
    print("\nConclusión: Hay evidencia significativa de asociación entre el genotipo RyR1 y el sexo.")
else:
    print("\nConclusión: No se encontró evidencia suficiente para afirmar que existe una asociación entre el genotipo RyR1 y el sexo.")
