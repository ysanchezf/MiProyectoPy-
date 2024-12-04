# Importar librerías necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Ruta al archivo CSV
file_path = r"C:\Users\Magia\OneDrive\Documentos\Mucha UNPHU kk\LP-A\FInal\train.csv"

# Cargar datos
data = pd.read_csv(file_path)

# Preprocesamiento
data_cleaned = data.drop(columns=['ADDRESS'])  # Eliminar columna irrelevante
data_cleaned = pd.get_dummies(data_cleaned, columns=['POSTED_BY', 'BHK_OR_RK'], drop_first=True)  # Codificar variables categóricas

# Separar características (X) y objetivo (y)
X = data_cleaned.drop(columns=['TARGET(PRICE_IN_LACS)'])
y = data_cleaned['TARGET(PRICE_IN_LACS)']

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelo de Regresión Lineal
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
y_pred_lin = lin_reg.predict(X_test_scaled)

# Evaluación de Regresión Lineal
mse_lin = mean_squared_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)

# Transformar objetivo para Regresión Logística
y_binary = (y > y.median()).astype(int)
y_train_bin = (y_train > y_train.median()).astype(int)
y_test_bin = (y_test > y_train.median()).astype(int)

# Modelo de Regresión Logística
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train_bin)
y_pred_log = log_reg.predict(X_test_scaled)

# Evaluación de Regresión Logística
accuracy_log = accuracy_score(y_test_bin, y_pred_log)
conf_matrix_log = confusion_matrix(y_test_bin, y_pred_log)

# Resultados formateados
print(f"\nResultados de Regresión Lineal:")
print(f"  - MSE: {mse_lin:.2f}")
print(f"  - R2: {r2_lin:.2f}")

print(f"\nResultados de Regresión Logística:")
print(f"  - Precisión: {accuracy_log:.2f}")
print(f"  - Matriz de Confusión:")
print(conf_matrix_log)

# Visualización Creativa

# Gráfico 1: Predicciones vs Precios Reales (Regresión Lineal)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lin, alpha=0.5, label="Predicciones vs Reales", color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Predicción Perfecta")
plt.xlabel("Precios Reales (en Lacs)")
plt.ylabel("Precios Predichos (en Lacs)")
plt.title("Regresión Lineal: Predicciones vs Precios Reales")
plt.legend()
plt.grid()
plt.show()

# Gráfico 2: Matriz de Confusión (Regresión Logística)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_log, annot=True, fmt="d", cmap="coolwarm", xticklabels=["Bajo", "Alto"], yticklabels=["Bajo", "Alto"])
plt.xlabel("Etiqueta Predicha")
plt.ylabel("Etiqueta Verdadera")
plt.title("Matriz de Confusión - Regresión Logística")
plt.show()

# Gráfico 3: Distribución de Precios
plt.figure(figsize=(10, 6))
sns.histplot(y, bins=50, kde=True, color="purple")
plt.axvline(y.median(), color='red', linestyle='--', label="Mediana de Precios")
plt.title("Distribución de los Precios")
plt.xlabel("Precio (en Lacs)")
plt.ylabel("Frecuencia")
plt.legend()
plt.grid()
plt.show()

# Gráfico 4: Importancia de las Características (Regresión Logística)
feature_importance = pd.Series(log_reg.coef_[0], index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(12, 8))
feature_importance.plot(kind='barh', color="orange")
plt.title("Importancia de las Características - Regresión Logística")
plt.xlabel("Valor del Coeficiente")
plt.ylabel("Características")
plt.grid()
plt.show()

# Tabla Resumen
results_summary = pd.DataFrame({
    "Métricas": ["MSE", "R2", "Precisión"],
    "Regresión Lineal": [mse_lin, r2_lin, None],
    "Regresión Logística": [None, None, accuracy_log]
})
print("\nResumen de Resultados:")
print(results_summary)
