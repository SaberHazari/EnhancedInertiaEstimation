import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

dataset = pd.read_excel('train_0.5s_delay.xlsx').values
train_percentage = 80

def slice_dataset(dataset, percentage):
    data_size = len(dataset)
    idx = np.arange(data_size)
    np.random.shuffle(idx)
    cut = int(data_size * percentage / 100)
    return dataset[idx,:][:cut], dataset[idx,:][cut:]

train_dataset, val_dataset = slice_dataset(dataset, train_percentage)

x_train = train_dataset[:,0:2600].astype(float)
Y_train = train_dataset[:,2600].astype(float)
x_test  = val_dataset[:,0:2600].astype(float)
Y_test  = val_dataset[:,2600].astype(float)

# Reshape
X_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
X_test  = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
Y_train = Y_train.reshape(-1,1)
Y_test  = Y_test.reshape(-1,1)

def pinn_loss(y_true, y_pred):
    mse_loss = keras.backend.mean(keras.backend.square(y_true - y_pred))
    penalty = keras.backend.mean(keras.backend.relu(-y_pred))
    return mse_loss + 0.01 * penalty

inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
x = Bidirectional(LSTM(32))(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.01)(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(1, activation='linear')(x)

pinn_model = Model(inputs=inputs, outputs=outputs)
pinn_model.compile(optimizer='adam', loss=pinn_loss, metrics=['mae','mse'])
pinn_model.summary()

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20)
tb_cb = TensorBoard(log_dir='log_PINN_5200', histogram_freq=1)

history = pinn_model.fit(X_train, Y_train,
                         validation_data=(X_test, Y_test),
                         epochs=200,
                         batch_size=16,
                         callbacks=[reduce_lr, tb_cb])

pred = pinn_model.predict(X_test)
mae = mean_absolute_error(Y_test, pred)
mse = mean_squared_error(Y_test, pred)
r2  = r2_score(Y_test, pred)

ERR = np.abs(pred - Y_test).reshape(-1)
def acc(thr): return np.mean(ERR <= thr)
tols = [0.1, 0.2, 0.3, 0.4, 0.5]
accs = [acc(t) for t in tols]

print(f"MAE: {mae:.6f}, MSE: {mse:.6f}, R²: {r2:.6f}")
for t, a in zip(tols, accs):
    print(f"Acc ≤ {t}: {a:.4f}")

# Learning Curve
plt.figure()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.yscale('log')
plt.xlabel('Epoch'); plt.ylabel('Loss (MSE, log scale)')
plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('Learning Curve - PINN - delay.png')
plt.show()

# Prediction Results
plt.figure(figsize=(6,6))
plt.scatter(Y_test, pred, alpha=0.6)
plt.xlabel('Actual Inertia (s)'); plt.ylabel('Predicted Intertia (s)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('Prediction Results - PINN - delay.png')
plt.show()

# Distribution of Absolute Errors
plt.figure(figsize=(7,5))
plt.hist(ERR, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel("Absolute Error"); plt.ylabel("Number of Cases")
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig('Distribution of Absolute Errors - PINN - delay.png')
plt.show()

# Absolute Error per case
plt.figure(figsize=(10,5))
plt.plot(ERR, marker='o', linestyle='-', alpha=0.6)
plt.xlabel("Case Index"); plt.ylabel("Absolute Error")
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig('Absolute Error per case - PINN - delay.png')
plt.show()

# Metrics Summary
plt.figure(figsize=(6,4)); plt.axis('off')
lines = [
    f"MAE: {mae:.6f}", f"MSE: {mse:.6f}", f"R²: {r2:.6f}"
] + [f"Acc ≤ {t}: {a:.4f}" for t,a in zip(tols, accs)]
plt.text(0.05,0.95,"PINN Bi-LSTM Metrics\n\n"+ "\n".join(lines), fontsize=12, va="top")
plt.savefig("Metrics Summary - PINN - delay.png", bbox_inches="tight", dpi=300)
plt.show()