import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Datos sintéticos: años de experiencia vs. sueldo mensual 
PENDIENTE_REAL = 2500
INTERCEPTO_REAL = 30000
RUIDO = 5000

rng = np.random.default_rng(42)
X = rng.uniform(0, 10, size=(100, 1)) # X debe ser 2D
y = PENDIENTE_REAL * X + INTERCEPTO_REAL + rng.normal(0, RUIDO, size=(100, 1))

# 2. Dividir en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Entrenar el Modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Como y es 2D, coef_ tiene forma (1, 1) y intercept_ forma(1,)
pendiente = modelo.coef_[0][0]
intercepto = modelo.intercept_[0]

print(f"Parámetros    Real    Estimado")
print(f"Pendiente  {PENDIENTE_REAL:>10.2f} {pendiente:>12.2f}")
print(f"Intercepto {INTERCEPTO_REAL:>10.2f} {intercepto:>12.2f}")

# 4. Evaluar en datos de prueba
y_pred = modelo.predict(X_test) 

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nMSE: {mse:,.2f}")
print(f"RMSE: {rmse:,.2f} (error típico de la predicción)")
print(f"R2: {r2:.3f}")

# 5. Visualización: ajuste + residuos
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Izquierda: datos y recta (sobre malla ordenada)
X_linea = np.linspace(0, 10, 100).reshape(-1, 1)
ax1.scatter(X_train, y_train, color="blue", alpha=0.5, label="Entrenamiento")
ax1.scatter(X_test, y_test, color="green", alpha=0.7, label="Prueba")
ax1.plot(X_linea, modelo.predict(X_linea), color="red", linewidth=2,
         label="Recta de regresión")
ax1.set_title("Regresión lineal")
ax1.set_xlabel("Años de exepriencia")
ax1.set_ylabel("sueldo")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.6)

# Derecha: de residuos (deberían verse aleatorios, centrados en 0)
residuos = (y_test - y_pred).ravel()
ax2.scatter(y_pred, residuos, color="purple", alpha=0.7)
ax2.axhline(0, color="red", linestyle="--")
ax2.set_title("Residuos vs. predicciones")
ax2.set_xlabel("Sueldo predicho")
ax2.set_ylabel("Residuo (real - predicho)")
ax2.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()