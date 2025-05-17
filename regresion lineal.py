import pandas as pd
import numpy as np


df = pd.read_excel('Lineal-data.xlsx')

X = df[['ID', 'Edad', 'Ingresos Mensuales', 'Horas Ejercicio x Semana', 'Nivel Educativo']].to_numpy()
X = np.hstack([np.ones((X.shape[0], 1)), X])

Y = df[['Índice Bienestar']].to_numpy()

Xt = X.T
XtX = Xt @ X
XtY = Xt @ Y
beta = np.linalg.inv(XtX) @ XtY

print("Coeficientes del modelo (β):")
print(f"Intercepto:                  {beta[0][0]:.4f}")
print(f"Edad:                        {beta[1][0]:.4f}")
print(f"Ingresos Mensuales:          {beta[2][0]:.4f}")
print(f"Horas Ejercicio x Semana:    {beta[3][0]:.4f}")
print(f"Nivel Educativo:             {beta[4][0]:.4f}")

nueva_persona = np.array([[52, 1.7, 6, 5]]) 
prediccion = nueva_persona @ beta

print(f"\nPredicción del Índice de Bienestar: {prediccion[0][0]:.2f} (escala 0-100)")
