from keras.layers import Layer
from keras import backend as K
import tensorflow as tf


class RoghLayer(Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(RoghLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.wl = self.add_weight(name='lower_weights',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        self.bl = self.add_weight(name='lower_bias',
                                  shape=(1, self.units),
                                  initializer='uniform',
                                  trainable=True)
        self.wu = self.add_weight(name='upper_weights',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        self.bu = self.add_weight(name='upper_bias',
                                  shape=(1, self.units),
                                  initializer='uniform',
                                  trainable=True)

        super(RoghLayer, self).build(input_shape)

    def call(self, inputs):
        net_l = tf.matmul(inputs, self.wl) + self.bl
        net_u = tf.matmul(inputs, self.wu) + self.bu
        if self.activation == None:
            ol = net_l
            ou = net_u
        elif self.activation == 'sigmoid':
            ol = tf.keras.activations.sigmoid(net_l)
            ou = tf.keras.activations.sigmoid(net_u)
        elif self.activation == 'tanh':
            ol = tf.keras.activations.tanh(net_l)
            ou = tf.keras.activations.tanh(net_u)
        elif self.activation == 'relu':
            ol = tf.keras.activations.relu(net_l)
            ou = tf.keras.activations.relu(net_u)

        o = (ol + ou) / 2
        return o

class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff,2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res


class FlexibleLayer(Layer):

    def __init__(self, units, activation=None, **kwargs):
        super(FlexibleLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.w = self.add_weight(name='weights',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        self.b = self.add_weight(name='bias',
                                  shape=(1, self.units),
                                  initializer='uniform',
                                  trainable=True)
        self.g = self.add_weight(name='flexible_parameter',
                                 shape=(1, self.units),
                                 initializer='uniform',
                                 trainable=True)

        super(FlexibleLayer, self).build(input_shape)


    def call(self, inputs):
        net = tf.matmul(inputs, self.wl) + self.bl
        if self.activation == None:
            o = net
        elif self.activation == 'sigmoid':
            net = tf.math.multiply(tf.math.abs(self.g), net)
            o = self.g * tf.keras.activations.sigmoid(net)
        elif self.activation == 'tanh':
            net = tf.math.multiply(tf.math.abs(self.g), net)
            o = tf.keras.activations.tanh(net)/self.g

        return o

