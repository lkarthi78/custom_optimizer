import tensorflow as tf

class LU_Optimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=1e-3, i_alpha=10, d_alpha=2,
                 steps_per_epoch=1, name="LU_Optimizer", **kwargs):
        super().__init__(learning_rate=learning_rate, name=name)

        self.steps_per_epoch = int(steps_per_epoch)
        self.i_alpha = tf.constant(float(i_alpha), tf.float32)
        self.d_alpha = tf.constant(float(d_alpha), tf.float32)

        self._lr = tf.Variable(float(learning_rate), dtype=tf.float32, trainable=False)

        self.prev_loss = tf.Variable(0.0, dtype=tf.float32, trainable=False)

        # graph-safe build flag
        self.slots_built = tf.Variable(False, dtype=tf.bool, trainable=False)

        # ---- dict only for indexing; lists hold slot variables ----
        self._idx = {}          # key -> index (Python dict)
        self._p_var = []        # list[tf.Variable] slot
        self._p_grad = []
        self._acc_grad = []
        self._grad = []
        self._acc_cnt = []   #per-variable accumulator count (scalar)

        # ---------------------------------------------------------------

    @property
    def learning_rate(self):
        return self._lr

    @learning_rate.setter
    def learning_rate(self, value):
        self._lr.assign(tf.cast(value, tf.float32))

    def _key(self, v):
        # Use .path if present, else fallback to name; normalize by stripping ":0"
        k = getattr(v, "path", None) or v.name
        return k.split(":")[0]

    def _safe(self, s):
        # Keras variable names cannot contain '/'
        return s.replace("/", "__").replace(":", "_")

    # list-based slot accessor
    def _slot(self, lst, v):
        return lst[self._idx[self._key(v)]]
                  
    def build(self, var_list):
        # Build slots exactly once per variable key
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
            self._acc_cnt.append(self.add_variable(
                shape=(), dtype=tf.float32, initializer="zeros", name=f"{sk}__acc_cnt"
            ))

            # init values
            self._p_var[i].assign(v)
            z = tf.zeros_like(v)
            self._p_grad[i].assign(z)
            self._acc_grad[i].assign(z)
            self._grad[i].assign(z)
            self._acc_cnt[i].assign(0.0)

        if len(self._p_var) > 0:
            self.slots_built.assign(True)

    def accumulate_epoch_grads(self, grads_and_vars, batch_in_epoch):
        # Reset accumulator on batch 0; accumulate each batch
        for g, v in grads_and_vars:
            acc = self._slot(self._acc_grad, v)
            cnt = self._slot(self._acc_cnt, v)

            def reset():
                acc.assign(tf.zeros_like(acc))
                cnt.assign(0.0)
                return 0

            tf.cond(tf.equal(batch_in_epoch, 0), reset, lambda: 0)

            # ---- MUST check None FIRST ----
            if g is None:
                continue

            if isinstance(g, tf.IndexedSlices):
                g = tf.convert_to_tensor(g)
            g = tf.cast(g, v.dtype)

            acc.assign_add(g)
            cnt.assign_add(1.0)

        # On last batch: average into grad slot
        def finalize_avg():
            for _, v in grads_and_vars:
                acc = self._slot(self._acc_grad, v)
                cnt = self._slot(self._acc_cnt, v)
                grd = self._slot(self._grad, v)

                denom = tf.maximum(cnt, 1.0)
                grd.assign(acc / tf.cast(denom, acc.dtype))
            return 0

        tf.cond(
            tf.equal(batch_in_epoch, self.steps_per_epoch - 1),
            finalize_avg,
            lambda: 0
        )

        # Move iterations
        self.iterations.assign_add(1)

    def update_var_graph(self, var_list, cur_loss, epoch_idx, total_epochs):

        cur_loss = tf.cast(cur_loss, tf.float32)

        epoch_idx = tf.cast(epoch_idx, tf.int64)
        total_epochs = tf.cast(total_epochs, tf.int64)

        def skip_last_epoch():
            return 0

        def do_update():
            def epoch0():
                self.prev_loss.assign(cur_loss)

                for v in var_list:
                    pv = self._slot(self._p_var, v)
                    pg = self._slot(self._p_grad, v)
                    gd = self._slot(self._grad, v)
                    pv.assign(v)
                    pg.assign(gd)

                # tf.print("\nE0_INIT | e=", epoch_idx," baseline_loss=", cur_loss," lr=", self._lr)

                penultimate = tf.equal(epoch_idx + 2, total_epochs)
                for v in var_list:
                    pv = self._slot(self._p_var, v)
                    pg = self._slot(self._p_grad, v)

                    tf.cond(
                        penultimate,
                        lambda pv=pv, vv=v: vv.assign(pv),
                        lambda pv=pv, pg=pg, vv=v: vv.assign(pv - self._lr * pg),
                    )
                return 0

            def epoch_ge_1():
                pl = self.prev_loss
                #pl_p = tf.identity(self.prev_loss)   # snapshot old prev_loss
                cl = cur_loss

                reject = tf.logical_or(
                    tf.logical_or(tf.math.is_nan(cl), tf.math.is_inf(cl)),
                    tf.logical_or(
                        tf.logical_and(cl >= 0.0, tf.logical_and(pl >= 0.0, cl > pl)),
                        tf.logical_or(
                            tf.logical_and(cl >= 0.0, tf.logical_and(pl <  0.0, cl > -pl)),
                            tf.logical_or(
                                tf.logical_and(cl <  0.0, tf.logical_and(pl >= 0.0, cl < -pl)),
                                tf.logical_and(cl <  0.0, tf.logical_and(pl <  0.0, cl < pl)),
                            )
                        )
                    )
                )

                def reject_branch():
                    self._lr.assign(self._lr / self.i_alpha)
                    #tf.print("\nREJECT | e=", epoch_idx, " cur=", cl, " prev=", pl_p, " lr->", self._lr)
                    return 0

                def accept_branch():
                    for v in var_list:
                        pv = self._slot(self._p_var, v)
                        pg = self._slot(self._p_grad, v)
                        gd = self._slot(self._grad, v)
                        pv.assign(v)
                        pg.assign(gd)

                    factor = (self.i_alpha * (self.d_alpha - 1.0) + 1.0) / self.d_alpha
                    self._lr.assign(self._lr * factor)
                    self.prev_loss.assign(cl)
                    #tf.print("\nACCEPT | e=", epoch_idx, " cur=", cl, " prev=", pl_p, " lr->", self._lr)
                    return 0

                tf.cond(reject, reject_branch, accept_branch)

                penultimate = tf.equal(epoch_idx + 2, total_epochs)
                for v in var_list:
                    pv = self._slot(self._p_var, v)
                    pg = self._slot(self._p_grad, v)
                    tf.cond(
                        penultimate,
                        lambda pv=pv, vv=v: vv.assign(pv),
                        lambda pv=pv, pg=pg, vv=v: vv.assign(pv - self._lr * pg),
                    )
                return 0

            return tf.cond(tf.equal(epoch_idx, 0), epoch0, epoch_ge_1)

        return tf.cond(tf.equal(epoch_idx + 1, total_epochs), skip_last_epoch, do_update)


class LUModel(tf.keras.Model):
    def __init__(self, *args, steps_per_epoch, total_epochs, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps_per_epoch = int(steps_per_epoch)
        self.total_epochs = int(total_epochs)

    def compile(self, optimizer, *args, **kwargs):
        super().compile(optimizer=optimizer, *args, **kwargs)

        # Align optimizer.steps_per_epoch with model.steps_per_epoch
        if hasattr(self.optimizer, "steps_per_epoch"):
            self.optimizer.steps_per_epoch = int(self.steps_per_epoch)

    def train_step(self, data):
        x, y = data

        batch_in_epoch = tf.math.floormod(self.optimizer.iterations, self.steps_per_epoch)
        epoch_idx = tf.math.floordiv(self.optimizer.iterations, self.steps_per_epoch)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred, training=True)

        invalid_loss = tf.logical_or(tf.math.is_nan(loss), tf.math.is_inf(loss))

        def abort_epoch0():
            tf.debugging.assert_all_finite(loss, "ABORT: Invalid loss at epoch 0.")
            return 0

        tf.cond(
            tf.logical_and(tf.equal(epoch_idx, 0), invalid_loss),
            abort_epoch0,
            lambda: 0
        )

        vars = self.trainable_variables

        def build_slots():
            self.optimizer.build(vars)
            return 0

        tf.cond(
            self.optimizer.slots_built,
            lambda: 0,
            build_slots
        )

        grads = tape.gradient(loss, vars)

        self.optimizer.accumulate_epoch_grads(list(zip(grads, vars)), batch_in_epoch)

        # Update metrics 
        for m in self.metrics:
            if "loss" in m.name:
                m.update_state(loss)
            else:
                m.update_state(y, y_pred)

        logs = {m.name: m.result() for m in self.metrics}

        def end_epoch():
            e_loss = logs["loss"]
            self.optimizer.update_var_graph(
                vars,
                cur_loss=e_loss,
                epoch_idx=epoch_idx,
                total_epochs=tf.constant(self.total_epochs, tf.int32),
            )
            return 0

        tf.cond(tf.equal(batch_in_epoch, self.steps_per_epoch - 1), end_epoch, lambda: 0)
        return logs
