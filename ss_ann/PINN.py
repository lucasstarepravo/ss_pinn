import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import math


class PINN_model:
    def __init__(self, input_size, num_neurons, num_layers, output_size):
        self.model = Sequential()
        # Input layer
        self.model.add(Dense(num_neurons, activation='relu', input_shape=(input_size,)))
        # Hidden layers
        for _ in range(1, num_layers):
            self.model.add(Dense(num_neurons, activation='relu'))
        # Output layer
        self.model.add(Dense(output_size, activation='linear'))

    def get_model(self):
        return self.model


def monomial_power(polynomial):
    """

    :param polynomial:
    :return:
    """
    monomial_exponent = [(total_polynomial - i, i)
                         for total_polynomial in range(1, polynomial + 1)
                         for i in range(total_polynomial + 1)]
    return np.array(monomial_exponent)


def calc_moments(neigh_xy_d, scaled_w, polynomial):
    mon_power = monomial_power(polynomial)
    monomial = []
    for power_x, power_y in mon_power:
        monomial.append((neigh_xy_d[:, :, 0] ** power_x * neigh_xy_d[:, :, 1] ** power_y) /
                        (math.factorial(power_x) * math.factorial(power_y)))
    moments = np.array(monomial) * scaled_w
    moments = np.sum(moments, axis=2)
    return moments.T


def physics_informed_loss(model, inputs, actual_outputs):

    predictions = model(inputs)


    data_loss = tf.reduce_mean(tf.square(actual_outputs - predictions))

    pred_moments = calc_moments(inputs, predictions, polynomial=2)
    target_moments = np.array((0, 0, 1, 0, 1))

    physics_loss = tf.reduce_mean(tf.square(target_moments - pred_moments))

    alpha = 5

    total_loss = data_loss + alpha*physics_loss

    return total_loss