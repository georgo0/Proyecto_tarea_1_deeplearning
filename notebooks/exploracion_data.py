import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1) Cargar dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

file_path = os.path.join(DATA_DIR, "dataset_limpio_sdss.csv")
df = pd.read_csv(file_path)

print("Shape dataset:", df.shape)

# 2) Convertir clase a numérica
# (solo para análisis de correlación)
df["class_num"] = df["class"].map({
    "GALAXY": 0,
    "STAR": 1,
    "QSO": 2
})

# 3) Matriz de correlación
corr_matrix = df.drop(columns=["class"]).corr()

# 4) Heatmap completo
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
plt.title("Matriz de correlación")
plt.show()

# 5) Correlación con la clase
corr_with_class = corr_matrix["class_num"].sort_values(ascending=False)

print("\nCorrelación con la clase:")
print(corr_with_class)

# 6) Detectar variables muy correlacionadas entre sí
threshold = 0.9  # puedes probar 0.85 también

print("\nPares de variables altamente correlacionadas (> 0.9):\n")

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            print(f"{col1} <--> {col2} : {corr_matrix.iloc[i, j]:.3f}")
