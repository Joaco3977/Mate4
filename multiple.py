#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

# Cargar datos
df = pd.read_csv('archive/players_21.csv')

# Creación de la matriz de variables independientes
X = df[['wage_eur', 'overall', 'potential']].values

# Creación de la matriz de la variable dependiente
Y = df['value_eur'].values

# Coloca una columna con unos como primera columna de la matriz
X = np.column_stack((np.ones(X.shape[0]), X))

# Imprimir dimensiones
print(f"Dimensión de X: {X.shape}")
print(f"Dimensión de Y: {Y.shape}")

# Calcula β^= (X^⊤ X)^−1 X^⊤ Y
XTX = np.dot(X.T, X)
XTX_inv = np.linalg.inv(XTX)
XTy = np.dot(X.T, Y)
beta = np.dot(XTX_inv, XTy)

# Imprimir intercepto y coeficientes obtenidos manualmente
print(f"Intercepto (manual): {beta[0]:.4f}")
print(f"Coeficientes (manual): {beta[1:]}")

# Hacer predicciones
y_pred = np.dot(X, beta)

# Calcular R²
r2 = r2_score(Y, y_pred)
print(f"Coeficiente de determinación (R²): {r2:.4f}")

# Obtener el coeficiente de correlación múltiple R:
R = r2**0.5
print(f'Coeficiente de Correlación Múltiple R: {R:.4f}')
