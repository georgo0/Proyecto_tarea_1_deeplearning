#esto es de test nomas
import pandas as pd

#df = pd.read_csv("data\dataset_limpio_sdss.csv")
df = pd.read_pickle("C:/Users/jogot/Downloads/df_spec_classprobs.pkl")

# Ver nombres de columnas
print(df.columns)

# Ver primeras 5 filas completas
print(df.head())

# Ver distribución de clases
print("\nDistribución de clases:")
print(df["class"].value_counts())