import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
# Función para agregar una fila de datos al diccionario de seguimiento


def add_row(data: dict, iteration, prev, new, fnew, error):
    data['iteration'].append(iteration)
    data['prev'].append(prev)
    data['new'].append(new)
    data['fnew'].append(fnew)
    data['error'].append(error)


# Cargar datos
df = pd.read_csv('archive/players_21.csv')

# Seleccionamos las características independientes (wage_eur, overall, potential)
X = df[['wage_eur', 'overall', 'potential']].values

# Seleccionamos la variable dependiente (value_eur)
Y = df['value_eur'].values

# Escalar X y Y
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

# Escalamos X y Y
X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y.reshape(-1, 1)).flatten()

# Agregar columna de unos para el intercepto
X_scaled = np.column_stack((np.ones(X_scaled.shape[0]), X_scaled))

# Función de costo (MSE)


def f(beta, X, Y):
    m = len(Y)
    return (1 / (2 * m)) * np.sum((np.dot(X, beta) - Y) ** 2)

# Gradiente de la función de costo


def derivf(beta, X, Y):
    m = len(Y)
    return (1 / m) * np.dot(X.T, (np.dot(X, beta) - Y))


# Inicialización
beta_prev = np.zeros(X_scaled.shape[1])  # Coeficientes iniciales (todos ceros)
tol = 1e-9  # Tolerancia
step = 0.001  # Tasa de aprendizaje
iteration = 1
max_iters = 10000  # Máximo número de iteraciones
errors = []  # Guardar los errores para graficar

# Diccionario para almacenar las últimas iteraciones
track_data = {'iteration': [], 'prev': [], 'new': [], 'fnew': [], 'error': []}

# Algoritmo de descenso por gradiente
f_prev = f(beta_prev, X_scaled, Y_scaled)
beta_new = beta_prev - step * derivf(beta_prev, X_scaled, Y_scaled)
f_new = f(beta_new, X_scaled, Y_scaled)
error = abs(f_new - f_prev)
errors.append(f_new)

# Agregar los valores iniciales al diccionario de seguimiento
add_row(track_data, iteration, beta_prev, beta_new, f_new, error)

# Iterar hasta que el error sea menor que la tolerancia o se alcance el máximo de iteraciones
while error > tol and iteration < max_iters:
    iteration += 1
    beta_prev, f_prev = beta_new, f_new
    beta_new = beta_prev - step * derivf(beta_prev, X_scaled, Y_scaled)
    f_new = f(beta_new, X_scaled, Y_scaled)
    error = abs(f_new - f_prev)
    errors.append(f_new)

    # Agregar los valores actuales al diccionario de seguimiento
    add_row(track_data, iteration, beta_prev, beta_new, f_new, error)

# Desescalar los coeficientes
# Desescalar coeficientes de las características
beta_descaled = beta_new[1:] / scaler_X.scale_
# Desescalar intercepto
intercept_descaled = beta_new[0] - \
    np.sum(beta_new[1:] * scaler_X.mean_ / scaler_X.scale_)

# Desescalar los coeficientes de Y
beta_descaled = beta_descaled * scaler_Y.scale_
intercept_descaled = intercept_descaled * scaler_Y.scale_ + scaler_Y.mean_

# Mostrar los coeficientes desescalados
print(f"\nResultados del Descenso por Gradiente:")
print(f"Intercepto desescalado: {intercept_descaled}")
print(f"Coeficientes desescalados: {beta_descaled}")

# Mostrar parámetros utilizados
print(f"\nParámetros utilizados:")
print(f"  Tasa de aprendizaje: {step}")
print(f"  Número de iteraciones: {iteration}")
