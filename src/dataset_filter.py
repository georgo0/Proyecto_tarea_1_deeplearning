#data_filter.py
import pandas as pd

#Dataset se extrajo desde: https://zenodo.org/records/3768398

# Cargar dataset
print("Cargando dataset...")
df = pd.read_pickle("C:/Users/jogot/Downloads/df_spec_classprobs.pkl") #"colocar en pd.read_pickle() la ubicacion en donde se descargo el dataset .pkl" 
print("Tamaño inicial:", df.shape)

# Eliminar duplicados
df = df.drop_duplicates(subset=["objid"])
print("Tamaño después de eliminar duplicados:", df.shape)

# Eliminar columnas innecesarias
drop_cols = [
    "objid", "specObjID",
    "run2d", "programname", "instrument", "survey",
    "class_pred", "prob_g", "prob_q", "prob_s", "prob_best"
]

df = df.drop(columns=drop_cols, errors="ignore")

# Manejo de valores nulos
print("\nValores nulos por columna:")
print(df.isnull().sum())

print("\nEliminando filas con valores nulos...")
df = df.dropna()

print("Tamaño después de eliminar nulos:", df.shape)

# Reducir columnas para un dataset más pequeño
use_reduced = True  #False para usar todo

if use_reduced:
    print("\nUsando dataset reducido: ")
    selected_cols = [
        "ra", "dec", "z",
        "cmod_u", "cmod_g", "cmod_r", "cmod_i", "cmod_z",
        "psf_u", "psf_g", "psf_r", "psf_i", "psf_z",
        "ext_u", "ext_g", "ext_r", "ext_i", "ext_z",
        "w1", "w2", "w3", "w4", "snmedian",
        "zwarning", "match_dist", "class"
    ]

    df = df[selected_cols]

else:
    print("\nUsando dataset completo...")

print("Tamaño final:", df.shape)

# Ver distribución de clases
print("\nDistribución de clases:")
print(df["class"].value_counts())

#  Mostrar primeras 5 filas
print("\nPrimeras filas:")
print(df.head())

# Guardar dataset limpio
df.to_csv("data\dataset_limpio_sdss.csv", index=False)

print("\n Dataset guardado como dataset_limpio_sdss.csv")
