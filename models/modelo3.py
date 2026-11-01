import numpy as np
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Input, Dropout, LeakyReLU
from keras.optimizers import SGD
from keras.regularizers import l2
import matplotlib.pyplot as plt


# Hiperparametros 
LEARNING_RATE = 0.001      #probar: 0.1 / 0.01 / 0.001
MOMENTUM = 0.5          # Momentum del optimizador Nesterov -> 0.5, 0.9
BATCH_SIZE = 256         
EPOCHS = 20           

#Regularizacion
DROPOUT_RATE_in = 0.3        # Porbar con 0.2/ 0.3/ 0.5
DROPOUT_RATE_mid = 0.5

# 1. Cargar datos
X_train = np.load("data/X_train.npy")
X_val   = np.load("data/X_val.npy")
X_test  = np.load("data/X_test.npy")

y_train = np.load("data/y_train.npy")
y_val   = np.load("data/y_val.npy")
y_test  = np.load("data/y_test.npy")

print("Datos cargados:")
print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)
# ================================

# Modelo 
model = Sequential([
    Input(shape=(X_train.shape[1],)),

    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(DROPOUT_RATE_in),

    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(DROPOUT_RATE_mid),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(DROPOUT_RATE_mid),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(DROPOUT_RATE_mid),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(DROPOUT_RATE_mid),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(DROPOUT_RATE_mid),

    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(DROPOUT_RATE_mid),

    Dense(3, activation='softmax')
])
# ================================

# Optimizador: Gradiente Estocastico + Nesterov
optimizer = SGD(
    learning_rate=LEARNING_RATE,
    momentum=MOMENTUM,
    nesterov=True
)

# Compilación
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Entrenamiento
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)


# Evaluación TESTING
print("\nEvaluando TESTING")
loss, acc = model.evaluate(X_test, y_test)

print(f"\nTest Accuracy: {acc:.4f}")
print(f"Test Error: {(1 - acc)*100:.2f}%")


# Calcular error 
train_error = [(1 - x)*100 for x in history.history['accuracy']]
val_error   = [(1 - x)*100 for x in history.history['val_accuracy']]


# Gráfico de LOSS y ACCURACY
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

plt.figure()
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

# Gráfico de ERROR (%)
plt.plot(train_error, label='train error (%)')
plt.plot(val_error, label='val error (%)')
plt.legend()
plt.title("Error (%)")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()