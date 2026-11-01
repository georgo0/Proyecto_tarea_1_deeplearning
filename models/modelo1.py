import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import RMSprop

# Hiperparametros
LEARNING_RATE = 0.001
BATCH_SIZE = 256
EPOCHS = 20

DROPOUT_RATE = 0.3

# Cargar datos
X_train = np.load("data/X_train.npy")
X_val   = np.load("data/X_val.npy")
X_test  = np.load("data/X_test.npy")

y_train = np.load("data/y_train.npy")
y_val   = np.load("data/y_val.npy")
y_test  = np.load("data/y_test.npy")

# Modelo
model = Sequential([
    Input(shape=(X_train.shape[1],)),

    Dense(128, activation='relu'),
    Dropout(DROPOUT_RATE),

    Dense(64, activation='relu'),
    Dropout(DROPOUT_RATE),

    Dense(32, activation='relu'),
    Dropout(DROPOUT_RATE),

    Dense(16, activation='relu'),

    Dense(3, activation='softmax')
])

# Compilacion
model.compile(
    optimizer=RMSprop(learning_rate=LEARNING_RATE),
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

# Evaluacion TEST
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Error: {(1 - test_acc)*100:.2f}%")

# Graficos
epochs_range = range(1, len(history.history['loss']) + 1)

# LOSS
plt.figure()
plt.plot(epochs_range, history.history['loss'], label='Train Loss')
plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
plt.axhline(y=test_loss, linestyle='--', label='Test Loss')

plt.legend()
plt.title("Loss (Train vs Validation vs Test)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# ACCURACY
plt.figure()
plt.plot(epochs_range, history.history['accuracy'], label='Train Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
plt.axhline(y=test_acc, linestyle='--', label='Test Accuracy')

plt.legend()
plt.title("Accuracy (Train vs Validation vs Test)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

# ERROR (%)
train_error = [(1 - x)*100 for x in history.history['accuracy']]
val_error   = [(1 - x)*100 for x in history.history['val_accuracy']]
test_error  = (1 - test_acc)*100

plt.figure()
plt.plot(epochs_range, train_error, label='Train Error (%)')
plt.plot(epochs_range, val_error, label='Validation Error (%)')
plt.axhline(y=test_error, linestyle='--', label='Test Error (%)')

plt.legend()
plt.title("Error (%)")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()