import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import classification_report
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

#  CONFIGURACIÓN PARA EXPERIMENTAR
#  Cambiar esto para ver comportamiento del modelo en base a la correlacion entre las columnas

#DROP_COLUMNS = ["z","psf_u", "psf_g", "psf_r", "psf_i", "psf_z","cmod_u", "cmod_g", "cmod_r", "cmod_i", "cmod_z"]  #-> #Dataset menos representativo

DROP_COLUMNS = ["z", "w1","ext_u", "ext_g", "ext_i", "ext_z","psf_u", "psf_g", "psf_i", "psf_z","cmod_u", "cmod_g", "cmod_i", "cmod_z"] #-> LA QUE SE OBTUVO POR ANALISIS

# 1) Cargar dataset
file_path = os.path.join(DATA_DIR, "dataset_limpio_sdss.csv")
df = pd.read_csv(file_path)

print("Tamaño inicial:", df.shape)

# 2) Eliminar columnas problemáticas
print("\nColumnas eliminadas:", DROP_COLUMNS)

df = df.drop(columns=DROP_COLUMNS, errors="ignore")

print("Tamaño después de eliminar columnas:", df.shape)

# 3) Separar variables
X = df.drop(columns=["class"])
y = df["class"]

# 4) Encoding
le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y)

print("Clases:", list(le.classes_))

# 5) Split (98-1-1)
# Test (1%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.01,
    stratify=y,
    random_state=42
)

# Validation (1%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.0101,
    stratify=y_temp,
    random_state=42
)

print("\nShapes antes de normalizar:")
print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)

# 6) Normalización del dataset
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print("\nDatos normalizados")
print("Media (train):", X_train.mean())
print("Std (train):", X_train.std())

# 7) Guardar datasets en carpeta ..\data

np.save(os.path.join(DATA_DIR, f"X_train.npy"), X_train)
np.save(os.path.join(DATA_DIR, f"X_val.npy"), X_val)
np.save(os.path.join(DATA_DIR, f"X_test.npy"), X_test)

np.save(os.path.join(DATA_DIR, f"y_train.npy"), y_train)
np.save(os.path.join(DATA_DIR, f"y_val.npy"), y_val)
np.save(os.path.join(DATA_DIR, f"y_test.npy"), y_test)

print(f"\nDatasets guardados")

# 8) Guardar encoder
with open(os.path.join(DATA_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

print("LabelEncoder guardado")

