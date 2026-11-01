#Este script permite ver un histograma de algunos paramertros -> es de prueba igual
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Cargar dataset
df = pd.read_csv("data\dataset_limpio_sdss.csv")

# Seleccionar algunas columnas (no todas, para no saturar)
cols = ["ra", "z", "cmod_g", "psf_r", "w1"]

# 1. Histogramas 
for col in cols:
    plt.figure()
    df[col].hist()
    plt.title(f"ANTES de normalización: {col}")
    plt.xlabel(col)
    plt.ylabel("Frecuencia")
    plt.show()

# 2. Normalizar (solo para visualización)
#scaler = StandardScaler()
#df_scaled = df.copy()
#df_scaled[cols] = scaler.fit_transform(df[cols])

# 3. Histogramas DESPUÉS de normalizar
#for col in cols:
 #   plt.figure()
  #  df_scaled[col].hist()
   # plt.title(f"DESPUÉS de normalización: {col}")
    #plt.xlabel(col)
    #plt.ylabel("Frecuencia")
    #plt.show()