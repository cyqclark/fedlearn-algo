# Copyright 2021 Fedlearn authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__doc__ = "Train helper module for vertical deep learning"

import tensorflow as tf

class VFDNNTrainHelper(object):
    """
    VFDNN Train Helper:
        including metrics, optimizer, data
    """
    def __init__(self,
                 data,
                 parameters,
                 is_active=False,
                 training=True):
        self.data = data
        self.is_active = is_active
        self.training = training
        if self.training:
            self.initialize_parameters(parameters)
            # some configs
            self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            self.optimizer = tf.keras.optimizers.Adam()
            self.train_loss = tf.keras.metrics.Mean(name='train_loss')
            self.train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
        return None

    # data and parameters related
    def initialize_parameters(self, parameters):
        self.batch_size = parameters["batch_size"]
        self.num_epoch = parameters["num_epoch"]
        # TODO: change data loading
        if self.is_active and self.training:
            x_train = self.data["x"]
            y_train = self.data["y"]
            self.train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(self.batch_size)
        else:
            x_train = self.data["x"]
            self.train_ds = tf.data.Dataset.from_tensor_slices((x_train)).batch(self.batch_size)
        self.train_ds_iterator = iter(self.train_ds)
        self.end_of_epoch = False
        return None

    def get_batch(self):
        optional = self.train_ds_iterator.get_next_as_optional()
        if tf.reduce_all(optional.has_value()):
            self.last_batch = optional.get_value()
        else:
            self.end_of_epoch = True
            self.train_ds_iterator = iter(self.train_ds)
            self.last_batch = self.train_ds_iterator.get_next()
        return self.last_batch

    def get_last_batch(self):
        return self.last_batch

    def reset_epoch(self):
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.end_of_epoch = False

    def get_inference_data(self):
        return self.data['x']


