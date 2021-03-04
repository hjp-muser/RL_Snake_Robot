import tensorflow as tf
import numpy as np

from baselines.a2c.utils import fc
# from baselines.common.models import get_network_builder
from baselines.common.distributions import make_pdtype


def get_network_builder(network_type, num_layers=2, num_hidden=64, activation=tf.nn.tanh, layer_norm=False):
    def network_fn(X):
        h = tf.layers.flatten(X)
        for i in range(num_layers):
            h = fc(h, 'mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))
            if layer_norm:
                h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
            h = activation(h)

        return h
    if network_type == 'mlp':
        return network_fn
    else:
        raise NotImplementedError


class Network(object):
    def __init__(self, name, network='mlp', **network_kwargs):
        self.name = name

        # self.network_builder = get_network_builder(network)(**network_kwargs)
        def network_fn(X, num_layers=2, num_hidden=64, activation=tf.nn.leaky_relu, layer_norm=False):
            h = tf.layers.flatten(X)
            for i in range(num_layers):
                h = fc(h, 'mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))
                if layer_norm:
                    h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
                h = activation(h)

            return h
        if network == 'mlp':
            self.network_builder = network_fn
        else:
            raise NotImplementedError

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class PolicyNetwork(Network):
    def __init__(self, ac_space, name='policy', network='mlp', layer_norm=False, **network_kwargs):
        network_kwargs.update({"layer_norm": layer_norm})
        super().__init__(name=name, network=network, **network_kwargs)
        self.ac_space = ac_space
        self.pdtype = make_pdtype(ac_space)
        self.layer_norm = layer_norm

    def __call__(self, obs, reuse=True):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            policy_latent = self.network_builder(obs)
            action = tf.nn.tanh(fc(policy_latent, "pi/mean", self.ac_space.shape[0]))
            # action = fc(policy_latent, "pi/mean", self.ac_space.shape[0])

        return action


class QvalueNetwork(Network):
    def __init__(self, name='qf', network='mlp', layer_norm=False, **network_kwargs):
        network_kwargs.update({"layer_norm": layer_norm})
        super().__init__(name=name, network=network, **network_kwargs)
        self.layer_norm = layer_norm

    def __call__(self, obs, action, reuse=True):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            qf_input = tf.concat([obs, action], axis=-1)  # this assumes observation and action can be concatenated
            qf_latent = self.network_builder(qf_input)
            qf = fc(qf_latent, "last_fc", 1)
        return qf

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars


class ValueNetwork(Network):
    def __init__(self, name='vf', network='mlp', layer_norm=False, **network_kwargs):
        network_kwargs.update({"layer_norm": layer_norm})
        super().__init__(name=name, network=network, **network_kwargs)
        self.layer_norm = layer_norm

    def __call__(self, obs, reuse=True):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            vf_latent = self.network_builder(obs)
            vf = fc(vf_latent, "last_fc", 1)
        return vf

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
