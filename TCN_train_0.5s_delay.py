import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow import keras
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from tcn import TCN
import os

# Load Dataset
dataset = pd.read_excel('train_0.5s_delay.xlsx').values

train_percentage = 80
val_percentage = 20

def slice_dataset(dataset, percentage):
    data_size = len(dataset)
    index = np.arange(data_size)
    np.random.shuffle(index)
    dataset_random = dataset[index,:]
    return dataset_random[:int(data_size*percentage/100)], dataset_random[int(data_size*percentage/100):]

train_dataset, val_dataset = slice_dataset(dataset, train_percentage)

# Prepare Data
x_train = np.expand_dims(train_dataset[:, 0:2600].astype(float), axis=2)
Y_train = np.expand_dims(train_dataset[:, 2600].astype(float), axis=1)
x_test = np.expand_dims(val_dataset[:, 0:2600].astype(float), axis=2)
Y_test = np.expand_dims(val_dataset[:, 2600].astype(float), axis=1)

# Reshape
X_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
X_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

# Build TCN Model
inputs = keras.Input(shape=(X_train.shape[1], X_train.shape[2]))
x = TCN()(inputs)
x = keras.layers.Dense(128, activation='relu')(x)
x = keras.layers.Dropout(0.01)(x)
x = keras.layers.Dense(64, activation='relu')(x)
x = keras.layers.Dense(64, activation='relu')(x)
outputs = keras.layers.Dense(1, activation='linear')(x)

tcn_model = keras.Model(inputs=inputs, outputs=outputs)
tcn_model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
tcn_model.summary()

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, mode='auto')
log_filepath = os.path.join('log')
tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=1)

# Train TCN
history = tcn_model.fit(X_train, Y_train,
                        validation_data=(X_test, Y_test),
                        epochs=200,
                        batch_size=16,
                        callbacks=[reduce_lr, tb_cb])

# Evaluate
scores = tcn_model.evaluate(X_test, Y_test, verbose=0)
print('%s: %.2f%%' % (tcn_model.metrics_names[1], scores[1] * 100))

# Plot Loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.yscale('log')
plt.legend()
plt.savefig('Learning Curve - TCN - delay.png')
plt.show()

# Predictions & Error
predicted = tcn_model.predict(X_test)

plt.scatter(Y_test, predicted)
plt.xlabel('Actual Inertia (s)')
plt.ylabel('Predicted Inertia (s)')
plt.savefig('Prediction Results - TCN - delay.png')
plt.show()

mse = mean_squared_error(Y_test, predicted)
r2 = r2_score(Y_test, predicted)

print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"R² Score: {r2:.6f}")

ERR = np.abs(predicted - Y_test)
absvalues = ERR.reshape(-1)

plt.figure(figsize=(7,5))
plt.hist(ERR, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel("Absolute Error")
plt.ylabel("Number of Cases")
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig('Distribution of Absolute Errors - TCN - delay.png')
plt.show()

plt.figure(figsize=(10,5))
plt.plot(ERR, marker='o', linestyle='-', alpha=0.6)
plt.xlabel("Case Index")
plt.ylabel("Absolute Error")
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig('Absolute Error per case -TCN - delay.png')
plt.show()

def acc(threshold):
    return np.sum(absvalues <= threshold)/len(absvalues)

for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
    print(f'Tolerance={t}: Accuracy={acc(t)}')

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
acc_text = "\n".join([f"Accuracy @ {t}: {acc(t):.4f}" for t in thresholds])

metrics_text = (
    f"MSE: {mse:.4f}\n"
    f"R²: {r2:.4f}\n"
    f"{acc_text}"
)

plt.text(0.05, 0.9, metrics_text, fontsize=12, va="top")
plt.title("Model Performance Metrics", fontsize=14, weight="bold")
plt.savefig("Metrics Summary - TCN - delay.png", bbox_inches="tight", dpi=300)
plt.show()