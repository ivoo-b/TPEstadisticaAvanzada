
# Problema 5 - Asociación entre tipo de paciente y antecedentes familiares de cáncer prostático

import pandas as pd
import scipy.stats as stats

# Cargar los datos
ruta = r"microARN.csv"
datos = pd.read_csv(ruta)

# Nos aseguramos de que existan las columnas necesarias
if 'antecedentes' not in datos.columns or 'grupo' not in datos.columns:
    raise ValueError("Las columnas 'antecedentes' y 'grupo' deben estar presentes en el CSV")

# Tabla de contingencia
tabla_contingencia = pd.crosstab(datos['grupo'], datos['antecedentes'])

print("Tabla de contingencia grupo vs antecedentes familiares:")
print(tabla_contingencia)

# Verificar que todos los valores esperados sean >= 5 para aplicar chi-cuadrado
chi2, p, dof, expected = stats.chi2_contingency(tabla_contingencia)
print("\nFrecuencias esperadas bajo independencia:")
print(pd.DataFrame(expected, index=tabla_contingencia.index, columns=tabla_contingencia.columns))

# Decisión según valor p
print("\nResultado de la prueba chi-cuadrado:")
print(f"Estadístico: {chi2:.4f}")
print(f"Valor p: {p:.4f}")

if p < 0.05:
    print("\nConclusión: Existe una asociación estadísticamente significativa entre el tipo de paciente y los antecedentes familiares.")
else:
    print("\nConclusión: No se encontró asociación estadísticamente significativa entre el tipo de paciente y los antecedentes familiares.")
