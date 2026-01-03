import numpy as np
import tensorflow as tf
from LU_M_Optimizer import LU_Optimizer, LUModel

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

'''
y_train = np.array([
 0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,
 0,0,0,0,1,0,0,1,0,0,0,1,1,0,0,0,0,1,1,0,1,0,0,1,
 1,1
], dtype=np.float32).reshape(-1, 1)
'''

d0 = 30.0
y_train = (np.abs(x_train.reshape(-1) - 50.0) >= d0).astype(np.float32).reshape(-1,1)


N = x_train.shape[0]
steps_per_epoch = 1    
batch_size =int(np.ceil(N / steps_per_epoch))
epochs = 50

# Optional (recommended): scale x
x_train_s = (x_train - 50.0) / 50.0

# model layers unchanged
inputs = tf.keras.Input(shape=(1,), dtype=tf.float32)
x = tf.keras.layers.Dense(8, activation="relu", name="L1")(inputs)
x = tf.keras.layers.Dense(16, activation="relu", name="L2")(x)
x = tf.keras.layers.Dense(8, activation="relu", name="L3")(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="L4")(x)

steps_per_epoch = 1
epochs = 50

model = LUModel(inputs=inputs, outputs=outputs,
                steps_per_epoch=steps_per_epoch, total_epochs=epochs)

opt = LU_Optimizer(learning_rate=1e-3, i_alpha=10, d_alpha=2, steps_per_epoch=steps_per_epoch)

model.compile(
    optimizer=opt,
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy(name="acc")],
    run_eagerly=True  # REQUIRED for numpy()
)

model.fit(x_train_s, y_train, epochs=epochs, batch_size=x_train_s.shape[0], steps_per_epoch=steps_per_epoch)

x_test = np.array([[46],[6],[51],[48],[98]], dtype=np.float32)
x_test_s = (x_test - 50.0) / 50.0
yhat = model.predict(x_test_s, verbose=0)
print("x_test:", x_test.reshape(-1))
print("yhat:", yhat.reshape(-1))
print("pred labels:", (yhat.reshape(-1) >= 0.5).astype(int))
