import numpy as np
import tensorflow as tf


class LU_Optimizer(tf.keras.optimizers.Optimizer):
    """
    Eager-only (run_eagerly=True) version.
    Uses .numpy() / Python control flow for decisions.
    """

    def __init__(self, learning_rate=1e-3, i_alpha=10, d_alpha=2,
                 steps_per_epoch=1, name="LU_Optimizer", **kwargs):
        super().__init__(learning_rate=learning_rate, name=name)

        self.steps_per_epoch = int(steps_per_epoch)
        self.i_alpha = float(i_alpha)
        self.d_alpha = float(d_alpha)

        self._lr = tf.Variable(float(learning_rate), dtype=tf.float32, trainable=False)

        self.prev_loss = tf.Variable(0.0, dtype=tf.float32, trainable=False)

        self.slots_built = False  # Python flag (eager-only)

        # key -> index (Python dict); lists hold slot vars
        self._idx = {}
        self._p_var = []
        self._p_grad = []
        self._acc_grad = []
        self._grad = []

    @property
    def learning_rate(self):
        return self._lr

    @learning_rate.setter
    def learning_rate(self, value):
        self._lr.assign(tf.cast(value, tf.float32))

    def _key(self, v):
        k = getattr(v, "path", None) or v.name
        return k.split(":")[0]

    def _safe(self, s):
        return s.replace("/", "__").replace(":", "_")

    def _slot(self, lst, v):
        return lst[self._idx[self._key(v)]]

    def build(self, var_list):
        # Build slots exactly once per variable key (eager)
        for v in var_list:
            k = self._key(v)
            if k in self._idx:
                continue

            sk = self._safe(k)
            i = len(self._p_var)
            self._idx[k] = i

            self._p_var.append(self.add_variable(
                shape=v.shape, dtype=v.dtype, initializer="zeros", name=f"{sk}__p_var"
            ))
            self._p_grad.append(self.add_variable(
                shape=v.shape, dtype=v.dtype, initializer="zeros", name=f"{sk}__p_grad"
            ))
            self._acc_grad.append(self.add_variable(
                shape=v.shape, dtype=v.dtype, initializer="zeros", name=f"{sk}__acc_grad"
            ))
            self._grad.append(self.add_variable(
                shape=v.shape, dtype=v.dtype, initializer="zeros", name=f"{sk}__grad"
            ))

            # init values
            self._p_var[i].assign(v)
            z = tf.zeros_like(v)
            self._p_grad[i].assign(z)
            self._acc_grad[i].assign(z)
            self._grad[i].assign(z)

        if len(self._p_var) > 0:
            self.slots_built = True

    def accumulate_epoch_grads(self, grads_and_vars, batch_in_epoch):
        # Eager: reset accumulator on batch 0; accumulate each batch
        batch_in_epoch = int(batch_in_epoch.numpy()) if tf.is_tensor(batch_in_epoch) else int(batch_in_epoch)

        for g, v in grads_and_vars:
            if isinstance(g, tf.IndexedSlices):
                g = tf.convert_to_tensor(g)
            g = tf.cast(g, v.dtype)

            acc = self._slot(self._acc_grad, v)
            if batch_in_epoch == 0:
                acc.assign(tf.zeros_like(acc))
            acc.assign_add(g)

        # On last batch: average into grad slot
        if batch_in_epoch == self.steps_per_epoch - 1:
            denom = float(self.steps_per_epoch)
            for g, v in grads_and_vars:
                acc = self._slot(self._acc_grad, v)
                grd = self._slot(self._grad, v)
                grd.assign(acc / tf.cast(denom, acc.dtype))

        # Move iterations
        self.iterations.assign_add(1)

    def update_var_graph(self, var_list, cur_loss, epoch_idx, total_epochs):
        # Eager numpy/python logic
        if tf.is_tensor(cur_loss):
            cl = float(cur_loss.numpy())
        else:
            cl = float(cur_loss)

        epoch_idx = int(epoch_idx.numpy()) if tf.is_tensor(epoch_idx) else int(epoch_idx)
        total_epochs = int(total_epochs.numpy()) if tf.is_tensor(total_epochs) else int(total_epochs)

        # skip last epoch
        if (epoch_idx + 1) == total_epochs:
            return 0

        # epoch 0
        if epoch_idx == 0:
            self.prev_loss.assign(cl)

            for v in var_list:
                pv = self._slot(self._p_var, v)
                pg = self._slot(self._p_grad, v)
                gd = self._slot(self._grad, v)
                pv.assign(v)
                pg.assign(gd)

            penultimate = ((epoch_idx + 2) == total_epochs)
            for v in var_list:
                pv = self._slot(self._p_var, v)
                pg = self._slot(self._p_grad, v)
                if penultimate:
                    v.assign(pv)
                else:
                    v.assign(pv - self._lr * pg)
            return 0

        # epoch >= 1
        pl = float(self.prev_loss.numpy())

        reject = (
            (np.isnan(cl) or np.isinf(cl)) or
            ((cl >= 0.0 and pl >= 0.0 and cl > pl) or
             (cl >= 0.0 and pl <  0.0 and cl > -pl) or
             (cl <  0.0 and pl >= 0.0 and cl < -pl) or
             (cl <  0.0 and pl <  0.0 and cl < pl))
        )

        if reject:
            self._lr.assign(self._lr / self.i_alpha)
        else:
            for v in var_list:
                pv = self._slot(self._p_var, v)
                pg = self._slot(self._p_grad, v)
                gd = self._slot(self._grad, v)
                pv.assign(v)
                pg.assign(gd)

            factor = (self.i_alpha * (self.d_alpha - 1.0) + 1.0) / self.d_alpha
            self._lr.assign(self._lr * factor)
            self.prev_loss.assign(cl)

        penultimate = ((epoch_idx + 2) == total_epochs)
        for v in var_list:
            pv = self._slot(self._p_var, v)
            pg = self._slot(self._p_grad, v)
            if penultimate:
                v.assign(pv)
            else:
                v.assign(pv - self._lr * pg)

        return 0


class LUModel(tf.keras.Model):
    def __init__(self, *args, steps_per_epoch, total_epochs, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps_per_epoch = int(steps_per_epoch)
        self.total_epochs = int(total_epochs)

    def compile(self, optimizer, *args, **kwargs):
        # IMPORTANT: run_eagerly=True for numpy()-based logic
        kwargs.setdefault("run_eagerly", True)
        super().compile(optimizer=optimizer, *args, **kwargs)

        if hasattr(self.optimizer, "steps_per_epoch"):
            self.optimizer.steps_per_epoch = int(self.steps_per_epoch)

    def train_step(self, data):
        x, y = data

        # NOTE: optimizer.iterations is a tf.Variable; use .numpy() for eager logic
        it = int(self.optimizer.iterations.numpy())
        batch_in_epoch = it % self.steps_per_epoch
        epoch_idx = it // self.steps_per_epoch

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred, training=True)

        # Abort if invalid loss at epoch 0
        if epoch_idx == 0:
            lv = float(loss.numpy())
            if np.isnan(lv) or np.isinf(lv):
                tf.debugging.assert_all_finite(loss, "ABORT: Invalid loss at epoch 0.")

        vars_ = self.trainable_variables

        # Build slots once (eager)
        if not self.optimizer.slots_built:
            self.optimizer.build(vars_)

        grads = tape.gradient(loss, vars_)
        self.optimizer.accumulate_epoch_grads(list(zip(grads, vars_)), batch_in_epoch)

        # Metrics
        for m in self.metrics:
            if "loss" in m.name:
                m.update_state(loss)
            else:
                m.update_state(y, y_pred)

        logs = {m.name: m.result() for m in self.metrics}

        # End-of-epoch update
        if batch_in_epoch == self.steps_per_epoch - 1:
            e_loss = logs["loss"]
            self.optimizer.update_var_graph(
                vars_,
                cur_loss=e_loss,
                epoch_idx=epoch_idx,
                total_epochs=tf.constant(self.total_epochs, tf.int32),
            )

        return logs
