import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Load Dataset
dataset = pd.read_excel('train_3rd_wrv_SNR45db.xlsx').values

train_percentage = 80
val_percentage = 20


def slice_dataset(dataset, percentage):
    data_size = len(dataset)
    index = np.arange(data_size)
    np.random.shuffle(index)
    dataset_random = dataset[index, :]
    return dataset_random[:int(data_size * percentage / 100)], dataset_random[int(data_size * percentage / 100):]


train_dataset, val_dataset = slice_dataset(dataset, train_percentage)

# Prepare Data
x_train = np.expand_dims(train_dataset[:, 0:7000].astype(float), axis=2)
Y_train = np.expand_dims(train_dataset[:, 7000].astype(float), axis=1)
x_test = np.expand_dims(val_dataset[:, 0:7000].astype(float), axis=2)
Y_test = np.expand_dims(val_dataset[:, 7000].astype(float), axis=1)

# Reshape
X_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
X_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

# Build TCN
from keras.layers import Conv1D, Dense, Dropout, Flatten, Input
from keras.models import Model


def build_tcn(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv1D(filters=64, kernel_size=2, padding='causal', dilation_rate=1, activation='relu')(inputs)
    x = Conv1D(filters=64, kernel_size=2, padding='causal', dilation_rate=2, activation='relu')(x)
    x = Conv1D(filters=64, kernel_size=2, padding='causal', dilation_rate=4, activation='relu')(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.01)(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


tcn_model = build_tcn((X_train.shape[1], X_train.shape[2]))
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
predicted = tcn_model.predict(X_test)

mae = mean_absolute_error(Y_test, predicted)
mse = mean_squared_error(Y_test, predicted)
r2 = r2_score(Y_test, predicted)

ERR = np.abs(predicted - Y_test)
absvalues = ERR.reshape(-1)

plt.figure(figsize=(7,5))
plt.hist(ERR, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel("Absolute Error")
plt.ylabel("Number of Cases")
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig('Distribution of Absolute Errors - TCN - 45dB.png')
plt.show()

plt.figure(figsize=(10,5))
plt.plot(ERR, marker='o', linestyle='-', alpha=0.6)
plt.xlabel("Case Index")
plt.ylabel("Absolute Error")
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig('Absolute Error per case -TCN - 45dB.png')
plt.show()

def acc(threshold):
    return np.sum(absvalues <= threshold) / len(absvalues)


tolerances = [0.1, 0.2, 0.3, 0.4, 0.5]
accuracy_list = [acc(t) for t in tolerances]

print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"R² Score: {r2:.6f}")
print("Tolerance-based Accuracies:")
for t, a in zip(tolerances, accuracy_list):
    print(f"  Tolerance ≤ {t}: Accuracy = {a:.4f}")

# Plot Loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.yscale('log')
plt.legend()
plt.savefig('Learning Curve - TCN - 45dB.png')
plt.show()

plt.scatter(Y_test, predicted)
plt.xlabel('Actual Inertia (s)')
plt.ylabel('Predicted Inertia (s)')
plt.savefig('Prediction Results - TCN - 45dB.png')
plt.show()

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
acc_text = "\n".join([f"Accuracy @ {t}: {acc(t):.4f}" for t in thresholds])

metrics_text = (
    f"MSE: {mse:.4f}\n"
    f"R²: {r2:.4f}\n"
    f"{acc_text}"
)

plt.text(0.05, 0.9, metrics_text, fontsize=12, va="top")
plt.title("Model Performance Metrics", fontsize=14, weight="bold")
plt.savefig("Metrics Summary - TCN - 45dB.png", bbox_inches="tight", dpi=300)
plt.show()