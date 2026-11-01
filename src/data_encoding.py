import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# 1.Cargar dataset limpio
df = pd.read_csv("data\dataset_limpio_sdss.csv")

print("Clases originales:")
print(df["class"].value_counts())

# 2.Separar variables
X = df.drop(columns=["class"]) #respecto a la etiqueta 'class={GALAXY, STAR, QSO}'
y = df["class"]

# 3.Label Encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("\nClases transformadas:")
for i, clase in enumerate(le.classes_):
    print(f"{clase} → {i}")

# 4.One-Hot Encoding para las etiqueta clase Y
y_onehot = to_categorical(y_encoded)

print("\nPrimeras etiquetas codificadas:")
print(y_onehot[:5])

print("\nForma de y:", y_onehot.shape)