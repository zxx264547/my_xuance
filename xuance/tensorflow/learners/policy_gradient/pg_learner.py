"""
Policy Gradient (PG)
Paper link: https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf
Implementation: TensorFlow2
"""
from argparse import Namespace
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.learners import Learner


class PG_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module):
        super(PG_Learner, self).__init__(config, policy)
        if ("macOS" in self.os_name) and ("arm" in self.os_name):  # For macOS with Apple's M-series chips.
            if self.distributed_training:
                with self.policy.mirrored_strategy.scope():
                    self.optimizer = tk.optimizers.legacy.Adam(config.learning_rate)
            else:
                self.optimizer = tk.optimizers.legacy.Adam(config.learning_rate)
        else:
            if self.distributed_training:
                with self.policy.mirrored_strategy.scope():
                    self.optimizer = tk.optimizers.Adam(config.learning_rate)
            else:
                self.optimizer = tk.optimizers.Adam(config.learning_rate)
        self.ent_coef = config.ent_coef

    @tf.function
    def forward_fn(self, obs_batch, act_batch, ret_batch):
        with tf.GradientTape() as tape:
            self.policy(obs_batch)
            a_dist = self.policy.actor.dist
            log_prob = a_dist.log_prob(act_batch)

            a_loss = -tf.reduce_mean(ret_batch * log_prob)
            e_loss = tf.reduce_mean(a_dist.entropy())

            loss = a_loss - self.ent_coef * e_loss
            gradients = tape.gradient(loss, self.policy.trainable_variables)

            if self.use_grad_clip:
                self.optimizer.apply_gradients([
                    (tf.clip_by_norm(grad, self.grad_clip_norm), var)
                    for (grad, var) in zip(gradients, self.policy.trainable_variables)
                    if grad is not None
                ])
            else:
                self.optimizer.apply_gradients([(grad, var) for (grad, var) in
                                                zip(gradients, self.policy.trainable_variables) if grad is not None])
        return a_loss, e_loss

    @tf.function
    def learn(self, *inputs):
        if self.distributed_training:
            a_loss, e_loss = self.policy.mirrored_strategy.run(self.forward_fn, args=inputs)
            return (self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, a_loss, axis=None),
                    self.policy.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, e_loss, axis=None))
        else:
            return self.forward_fn(*inputs)

    def update(self, **samples):
        self.iterations += 1
        obs_batch = samples['obs']
        act_batch = samples['actions']
        ret_batch = samples['returns']

        a_loss, e_loss = self.learn(obs_batch, act_batch, ret_batch)

        info = {
            "actor-loss": a_loss.numpy(),
            "entropy": e_loss.numpy()
        }

        return info
