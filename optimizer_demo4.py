import numpy as np
import tensorflow as tf
from LU_G_Optimizer import LU_Optimizer, LUModel  # graph-safe versions

# -----------------------------
# Data
# -----------------------------
x_train = np.array([
 [16.],[92.],[21.],[26.],[40.],[16.],[78.],[24.],[22.],[ 5.],
 [38.],[35.],[64.],[17.],[23.],[ 5.],[74.],[67.],[23.],[91.],
 [28.],[62.],[56.],[ 1.],[82.],[44.],[65.],[51.],[ 5.],[29.],
 [77.],[ 5.],[43.],[23.],[74.],[88.],[92.],[75.],[27.],[17.],
 [22.],[ 9.],[89.],[25.],[ 3.],[56.],[66.],[94.],[ 2.],[12.]
], dtype=np.float32)

y_train = np.array([
  0.01,-0.06, 0.04, 0.07, 0.12, 0.01, 0.05, 0.06, 0.05,-0.07,
  0.11, 0.10, 0.11, 0.01, 0.05,-0.08, 0.07, 0.09, 0.05,-0.04,
  0.08, 0.11, 0.12,-0.12, 0.02, 0.12, 0.10, 0.12,-0.08, 0.08,
  0.05,-0.07, 0.12, 0.05, 0.07,-0.02,-0.05, 0.06, 0.07, 0.02,
  0.05,-0.04,-0.03, 0.06,-0.10, 0.12, 0.10,-0.07,-0.10,-0.02
], dtype=np.float32).reshape(-1, 1)

N = x_train.shape[0]
steps_per_epoch = 1     
batch_size =int(np.ceil(N / steps_per_epoch))
epochs = 50

# Optional : scale x so the network doesn't fight huge magnitudes
x_train_s = (x_train - 50.0) / 50.0

# -----------------------------
# Model (linear output)
# -----------------------------
inputs = tf.keras.Input(shape=(1,), dtype=tf.float32)
x = tf.keras.layers.Dense(8, activation="relu", name="L1")(inputs)
x = tf.keras.layers.Dense(16, activation="relu", name="L2")(x)
x = tf.keras.layers.Dense(8, activation="relu", name="L3")(x)
outputs = tf.keras.layers.Dense(1, activation=None, name="L4")(x)  # <-- linear

model = LUModel(inputs=inputs, outputs=outputs,
                steps_per_epoch=steps_per_epoch, total_epochs=epochs)

opt = LU_Optimizer(learning_rate=1e-3, i_alpha=10, d_alpha=2,
                   steps_per_epoch=steps_per_epoch)

model.compile(
    optimizer=opt,
    loss=tf.keras.losses.MeanSquaredError(),
    run_eagerly=False
)

model.fit(x_train_s, y_train, epochs=epochs, batch_size=batch_size)

# -----------------------------
# Predict
# -----------------------------
x_test = np.array([[2],[52],[41],[91]], dtype=np.float32)
x_test_s = (x_test - 50.0) / 50.0
yhat = model.predict(x_test_s, verbose=0)
print("x_test:", x_test.reshape(-1))
print("yhat:", yhat.reshape(-1))

yhat_train = model.predict(x_train_s, verbose=0)
mse = np.mean((yhat_train - y_train)**2)
print("train MSE:", mse)
print("train y min/max:", y_train.min(), y_train.max())
print("pred y min/max:", yhat_train.min(), yhat_train.max())