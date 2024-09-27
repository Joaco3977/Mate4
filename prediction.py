import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import statsmodels.api as sm

# Cargar los datos
df = pd.read_csv('archive/players_21.csv')

X = df[['wage_eur']].values
Y = df['value_eur'].values

# Agregar una columna de unos para el término independiente (intercepto)
X = np.column_stack((np.ones(X.shape[0]), X))

# Ajustar el modelo de regresión lineal
X_stats = sm.add_constant(X)
model = sm.OLS(Y, X_stats)
results = model.fit()

# Obtener los valores predichos
y_pred = np.dot(X, results.params)

# Calcular el coeficiente de determinación (R²)
r2 = r2_score(Y, y_pred)
print(f"Coeficiente de determinación (R²): {r2:.4f}")

# Calcular el coeficiente de correlación lineal (r)
r, _ = pearsonr(df['wage_eur'], df['value_eur'])
print(f"Coeficiente de correlación (r): {r:.4f}")

# Obtener los intervalos de confianza al 95% para los coeficientes
conf_int = results.conf_int(alpha=0.05)

# Crear un DataFrame para los coeficientes y los intervalos de confianza
results_df = pd.DataFrame({
    'Coeficiente': results.params,
    'Límite Inferior': conf_int[:, 0],
    'Límite Superior': conf_int[:, 1]
})
print("Intervalos de confianza al 95% para los coeficientes:")
print(results_df)

# Calcular los residuos y el error estándar de los predichos
residuals = Y - y_pred
mse = np.mean(residuals**2)
se_pred = np.sqrt(mse)

# Calcular los intervalos de predicción al 95%
t_value = 1.96  # Para un nivel de confianza del 95%
intervalo_inferior = y_pred - t_value * se_pred
intervalo_superior = y_pred + t_value * se_pred

# Calcular la proporción de veces que el valor real supera la incertidumbre de predicción
supera_incertidumbre = np.sum(Y > intervalo_superior) / len(Y)

# Media de las predicciones
media_predicciones = np.mean(y_pred)

print(f"Proporción de veces que el valor real supera la incertidumbre de predicción: {
      supera_incertidumbre:.4f}")
print(f"Media de las predicciones: {media_predicciones:.4f}")

# Graficar los valores reales vs predicciones
plt.figure(figsize=(8, 6))
plt.scatter(Y, y_pred, color='blue', label='Predicción')
plt.plot([min(Y), max(Y)], [min(Y), max(Y)], color='red',
         linestyle='--', label='Línea ideal')
plt.fill_between(Y, intervalo_inferior, intervalo_superior,
                 color='gray', alpha=0.2, label='Intervalo de predicción (95%)')
plt.title('Valor real del mercado vs Predicción del modelo')
plt.xlabel('Valor real del mercado')
plt.ylabel('Valor predicho por el modelo')
plt.legend()
plt.grid(True)
plt.show()
