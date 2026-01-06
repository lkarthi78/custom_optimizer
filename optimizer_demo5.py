import numpy as np
import tensorflow as tf

from LU_Optimizer import LU_Optimizer, LossStoreCallback

# -----------------------------
# Data: x1, x2 in [0, 20]
# s = |x1^2 - x2^2| -> 3 grades
# -----------------------------
rng = np.random.default_rng(0)

N = 5000
x1 = rng.uniform(0.0, 20.0, size=(N, 1)).astype(np.float32)
x2 = rng.uniform(0.0, 20.0, size=(N, 1)).astype(np.float32)

x_train = np.concatenate([x1, x2], axis=1)   # shape (N,2)

# signal
s = np.abs(x1**2 - x2**2).reshape(-1)

# quantile thresholds → balanced grades
q_low, q_high = np.quantile(s, [1/3, 2/3])

y_train = np.full(N, 1, dtype=np.int32)   # medium
y_train[s <= q_low]  = 0                  # low
y_train[s >= q_high] = 2                  # high

print("Grade thresholds:", q_low, q_high)
print("Class counts:", np.bincount(y_train))


steps_per_epoch = 2
batch_size =int(np.ceil(N / steps_per_epoch))
epochs = 500

# scale inputs 
x_mean = x_train.mean(axis=0, keepdims=True)
x_std  = x_train.std(axis=0, keepdims=True) + 1e-7
x_train_s = (x_train - x_mean) / x_std

inputs = tf.keras.Input(shape=(2,), dtype=tf.float32)
x = tf.keras.layers.Dense(32, activation="relu", name="L1")(inputs)
x = tf.keras.layers.Dense(64, activation="relu", name="L2")(x)
x = tf.keras.layers.Dense(32, activation="relu", name="L3")(x)
outputs = tf.keras.layers.Dense(3, activation="softmax", name="Out")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

opt = LU_Optimizer(
    learning_rate=1e-3,
    i_alpha=10,
    d_alpha=1.5,
    steps_per_epoch=steps_per_epoch
)

cb = LossStoreCallback(optimizer=opt, total_epochs=epochs)

model.compile(
    optimizer=opt,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    run_eagerly=True
)

model.fit(
    x_train_s, y_train,
    epochs=epochs,
    batch_size=batch_size,
    steps_per_epoch=steps_per_epoch,
    callbacks=[cb]
)

x_test = np.array([
    [20.0,  18.0],   # |400-324| = 76   → medium
     [17.0, 4.0],   # |289-16| = 273 → high
    [8.0, 10.0],   # |64-100| = 36 → low
    [11.0, 6.0],  # |121-36| = 85 → medium
], dtype=np.float32)

s_test = np.abs(x_test[:,0]**2 - x_test[:,1]**2)

x_test_s = (x_test - x_mean) / x_std
probs = model.predict(x_test_s, verbose=0)
pred = np.argmax(probs, axis=1)

print("x_test:\n", x_test)
print("signal |x1^2-x2^2|:", s_test)
print("pred grades:", pred)
print("probs:\n", probs)
