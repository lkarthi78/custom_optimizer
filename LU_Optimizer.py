import tensorflow as tf
import math
import numpy as np

class LossStoreCallback(tf.keras.callbacks.Callback):
    def __init__(self, optimizer, total_epochs=10):
        super().__init__()
        self.opt = optimizer
        self.opt.total_epochs = total_epochs

    def on_epoch_begin(self, epoch, logs=None):
        self.opt.current_epoch = epoch

    def on_train_batch_begin(self, batch, logs=None):
        self.opt.current_batch = batch

    def on_epoch_end(self, epoch, logs=None):
        # fetch epoch loss
        if logs and "loss" in logs and logs["loss"] is not None:
            cur = logs["loss"]
            if tf.is_tensor(cur):
                cur = float(cur.numpy())
            else:
                cur = float(cur)
            self.opt.cur_loss = cur
        else:
            self.opt.cur_loss = None

        # epoch-0 baseline + ABORT if invalid
        if self.opt.current_epoch == 0:
            if (self.opt.cur_loss is None or
                math.isnan(self.opt.cur_loss) or
                math.isinf(self.opt.cur_loss)):
                print("ABORT: invalid epoch-0 loss:", self.opt.cur_loss)
                self.model.stop_training = True
                return

            # baseline loss
            self.opt.prev_loss = self.opt.cur_loss

        # revise vars based on epoch loss + stored epoch grads
        self.opt.update_var()


class LU_Optimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=1e-3, i_alpha=10, d_alpha=2,
                 steps_per_epoch=1, name="LU_Optimizer", **kwargs):
        super().__init__(learning_rate=learning_rate, name=name)

        self.current_batch = 0
        self.total_epochs = 0
        self.current_epoch = 0
        self.steps_per_epoch = int(steps_per_epoch)

        self.i_alpha = i_alpha
        self.d_alpha = d_alpha

        self.cur_loss = None
        self.prev_loss = None

        self._index = {}
        self._p_var = []
        self._p_grad = []
        self._var = []
        self._grad = []

        self._lr = float(learning_rate)
        self._built_once = False

    def build(self, var_list):
        super().build(var_list)
        if self._built_once:
            return
        self._built_once = True

        # reset containers
        self._index = {}
        self._p_var = []
        self._var = []

        n = len(var_list)
        self._p_var = [None] * n        
        self._p_grad = [None] * n
        self._grad = [None] * n

        for i, v in enumerate(var_list):
            key = self._var_key(v)
            self._index[key] = i
            self._var.append(v)       # variable refs
            w = v.numpy().copy()
            self._p_grad[i] = np.zeros_like(w)
            self._grad[i] = np.zeros_like(w)

    def update_var(self):
        # skip last epoch (no need to propose next move)
        if self.current_epoch + 1 == self.total_epochs:
            return

        # Reject test 
        reject = (
            self.cur_loss is None or
            math.isnan(self.cur_loss) or
            math.isinf(self.cur_loss) or
            (self.cur_loss >= 0 and self.prev_loss >= 0 and self.cur_loss > self.prev_loss) or
            (self.cur_loss >= 0 and self.prev_loss <  0 and self.cur_loss > -self.prev_loss) or
            (self.cur_loss <  0 and self.prev_loss >= 0 and self.cur_loss < -self.prev_loss) or
            (self.cur_loss <  0 and self.prev_loss <  0 and self.cur_loss < self.prev_loss)
        )

        if reject:
            self._lr /= self.i_alpha
        else:
            # accept -> update baseline weights+grads
            for i, v in enumerate(self._var):
                self._p_var[i] = v.numpy().copy()
                self._p_grad[i] = self._grad[i].copy()
            if self.current_epoch != 0:
                self._lr *= ((self.i_alpha * (self.d_alpha - 1) + 1) / self.d_alpha)
                self.prev_loss = self.cur_loss

        # apply next weights
        for i, v in enumerate(self._var):
            if self.current_epoch + 2 == self.total_epochs:
                v.assign(self._p_var[i])
            else:
                v.assign(self._p_var[i] - self._lr * self._p_grad[i])

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        pairs = [(g, v) for (g, v) in grads_and_vars]

        if not self._built_once:
            self.build([v for _, v in pairs])

        # accumulate grads for epoch
        for grad, var in pairs:
            if grad is None:
                continue
            if isinstance(grad, tf.IndexedSlices):
                grad = tf.convert_to_tensor(grad)

            i = self._index[self._var_key(var)]

            if self.current_batch == 0:
                self._grad[i].fill(0.0)

            self._grad[i] += grad.numpy()

            if self.current_batch == self.steps_per_epoch - 1:
                self._grad[i] /= float(self.steps_per_epoch)

        # keep Keras iteration counter moving
        self.iterations.assign_add(1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "i_alpha": self.i_alpha,
            "d_alpha": self.d_alpha,
            "steps_per_epoch": self.steps_per_epoch,
        })
        return cfg