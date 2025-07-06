
# Prueba de independencia - Problema 3 TP4

import pandas as pd
import scipy.stats as stats

# Crear tabla de contingencia con los datos del problema
# Filas: Tratamientos (I, II, III)
# Columnas: Estado (Preñada, No preñada)

# Datos
preñadas = [49, 32, 35]
total = [70, 40, 40]
no_preñadas = [total[i] - preñadas[i] for i in range(3)]

# Crear DataFrame
datos = pd.DataFrame({
    'Preñada': preñadas,
    'No Preñada': no_preñadas
}, index=['Sin tratamiento', 'Sincronización hormonal', 'Sincronización + Gonadotropina'])

print("Tabla de contingencia:")
print(datos)

# Prueba de chi-cuadrado
chi2, p, dof, expected = stats.chi2_contingency(datos)

print("\nResultados de la prueba de independencia:")
print(f"Estadístico chi-cuadrado: {chi2:.4f}")
print(f"Valor p: {p:.4f}")
print(f"Grados de libertad: {dof}")
print("\nFrecuencias esperadas bajo H0:")
print(pd.DataFrame(expected, index=datos.index, columns=datos.columns))

if p < 0.05:
    print("\nConclusión: Existe evidencia significativa de que la tasa de preñez depende del tratamiento.")
else:
    print("\nConclusión: No hay evidencia suficiente para decir que la tasa de preñez difiere entre tratamientos.")
