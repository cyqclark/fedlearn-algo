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

__doc__ = "Passive Models"

import os

import numpy
import pandas

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.models import Model

import TrainHelper

# TODO: hold model on java side
_DEEP_MODEL = (None, None, None) # (model_token, model, train_helper)
_DEEP_MODEL_TRAIN_DATA = None
_TRAIN_FILE = "/export/Data/federated-learning-client/fk_train_passive.csv"
_CONFIG = {}

def _LOAD_CONFIG():
    """
    Load config from config file
    """
    _CONFIG["base_path"] = os.getcwd() + "/"
_LOAD_CONFIG()


class PassiveMLPModel(Model):
    def __init__(self,
                 data,
                 parameters):
        super(PassiveMLPModel, self).__init__()
        # initialize data and parameters, this might be pulled out in the future
        self.data = data
        self.model_input_shape = self.data['x'].shape[1:]
        self.d1_passive = Dense(16, activation='relu')
        self.d2_passive = Dense(8, activation='relu')
        self.d3_passive = Dense(1)
        self.connect_layer = self.d3_passive
        self.connect_layer_name = self.connect_layer.name
        self.gradient_connect_layer = {"MatMul": None, "Bias": None}
        self.call(numpy.random.randn(*([1] + list(self.model_input_shape))), init=True)
        # some configs
        self.optimizer = tf.keras.optimizers.Adam()

    @tf.custom_gradient
    def connect_layer_call(self, x):
        def grad(dy, variables):
            return [[None], [self.gradient_connect_layer["MatMul"], self.gradient_connect_layer["Bias"]]]
        return self.connect_layer(x), grad

    def call(self, x, init=False):
        x = self.d1_passive(x)
        x = self.d2_passive(x)
        if init:
            return self.connect_layer_call(x)
        # attention: no last layer call in training and inference
        else:
            return x, self.connect_layer_call(x)

    def get_connect_layer_weights(self):
        return self.connect_layer.get_weights()

    def receive_connect_gradient(self, gradients):
        # TODO: get correct logic of gradient when using eager execution
        if tf.executing_eagerly():
            self.gradient_connect_layer["MatMul"] = gradients[0]#.numpy()
            self.gradient_connect_layer["Bias"] = gradients[1]#.numpy()
        else:
            for gradi in gradients:
            # TODO: get better logic
                if "MatMul" in gradi.name:
                    self.gradient_connect_layer["MatMul"] = gradi
                elif "Bias" in gradi.name:
                    self.gradient_connect_layer["Bias"] = gradi
                else:
                    raise ValueError("Unsupported gradient!")
        return None

    def get_config(self):
        config = {}
        config.update({"d1_passive": self.d1_passive,
                       "d2_passive": self.d2_passive,
                       "d3_passive": self.d3_passive,
                       "connect_layer": self.connect_layer,
                       "connect_layer_name": self.connect_layer_name,
                       "gradient_connect_layer": self.gradient_connect_layer,
                      })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def get_connect_layer_feature(passive_model, train_helper, training=True):
    if training:
        data = train_helper.get_batch()
        return passive_model(data, training=False)[0]
    else:
        data = train_helper.get_inference_data()
        return passive_model(data, training=False)[0]

#TODO: build with @tf.function
def train_step_passive(passive_model, train_helper, connect_layer_gradients):
    data = train_helper.get_last_batch()
    with tf.GradientTape() as tape_passive:
        _, output = passive_model(data, training=True)
        passive_model.receive_connect_gradient(connect_layer_gradients)
    gradients_passive = tape_passive.gradient(output, passive_model.trainable_variables)
    train_helper.optimizer.apply_gradients(zip(gradients_passive, passive_model.trainable_variables))
    return None

# TODO: optimize logic for train and inference
def get_or_assign_model_passive(model_token,
                                parameters=None,
                                data=None):
    global _DEEP_MODEL

    # get data
    if model_token == _DEEP_MODEL[0]:
        # model is being hold in _DEEP_MODEL
        model = _DEEP_MODEL[1]
        if data is None:
            train_helper = _DEEP_MODEL[2]
        else:
            # add new dataset, inference
            data = {"x": numpy.array(data["x"]).astype("float32")}
            train_helper = TrainHelper.VFDNNTrainHelper(
                data, parameters, is_active=False, training=False)
    elif os.path.exists(_CONFIG["base_path"] + model_token + "/"):
        # load model from path and create train helper
        model = keras.models.load_model(_CONFIG["base_path"] + model_token + "/")
        if data is None:
            data = get_data_passive()
        else:
            data = {"x": numpy.array(data["x"]).astype("float32")}
        train_helper = TrainHelper.VFDNNTrainHelper(
            data, parameters, is_active=False, training=False)
    else:
        # create train helper
        if data is None:
            data = get_data_passive()
        else:
            data = {"x": numpy.array(data["x"]).astype("float32")}
        train_helper = TrainHelper.VFDNNTrainHelper(
            data, parameters, is_active=False, training=True)
        # create and train a new model
        # get model type
        model_type = model_token.split("_")[-1]
        if model_type == "MLP":
            model = PassiveMLPModel(data, parameters)
        else:
            raise ValueError("Invalid model type!")
    _DEEP_MODEL = (model_token, model, train_helper)
    return model, train_helper


def get_data_passive(data=None):
    if data is None:
        g = pandas.read_csv(_TRAIN_FILE)
        return {"x": g.loc[:, g.columns[1:]].values.astype("float32")}
    else:
        raise NotImplementedError("Not implement yet!")


def save_model(model_token, model):
    model.save(_CONFIG["base_path"] + model_token + "_passive/")


def _train_step_test(passive_model, x_passive, connect_layer_gradients, optimizer):
    with tf.GradientTape() as tape_passive:
        _, output = passive_model(x_passive, training=True)
        passive_model.receive_connect_gradient(connect_layer_gradients)
    gradients_passive = tape_passive.gradient(output, passive_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients_passive, passive_model.trainable_variables))
    return None
