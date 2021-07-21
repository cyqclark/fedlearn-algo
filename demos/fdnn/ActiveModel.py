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

__doc__ = "Active model"

import os

import numpy
import pandas

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.models import Model

import TrainHelper

# TODO: hold model on java side
_DEEP_MODEL = (None, None, None) # (model_token, model, train_helper)
_TRAIN_FILE = None
_TRAIN_FILE = "/export/Data/federated-learning-client/fk_train_active.csv"
_CONFIG = {}

def _LOAD_CONFIG():
    """
    Load config from config file
    """
    _CONFIG["base_path"] = os.getcwd() + "/"
_LOAD_CONFIG()


class ActiveMLPModel(Model):
    def __init__(self, passive_connect_layer_weights):
        super(ActiveMLPModel, self).__init__()
        self.d1_active = Dense(16, activation='relu')
        self.d2_active = Dense(8, activation='relu')
        self.d3_active = Dense(1, activation='relu')
        self.d4 = Dense(1, activation='sigmoid')
        # initialize passive connect layer
        """
        self.flag_bypass_weights = False
        self.passive_connect_layer_weights = passive_connect_layer_weights
        """
        w = passive_connect_layer_weights[0]
        self.passive_connect_layer_name = "connect_layer"
        self.passive_connect_layer = Dense(w.shape[-1], name=self.passive_connect_layer_name)
        self.passive_connect_layer(numpy.random.randn(1, w.shape[0]))
        self.passive_connect_layer.set_weights(passive_connect_layer_weights)

        #self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        #self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    def call(self, inputs): # inputs: [x_active, x_passive]
        x_active, x_passive = inputs
        x_active = self.d1_active(x_active)
        x_active = self.d2_active(x_active)
        x_active = self.d3_active(x_active)
        x_passive = self.passive_connect_layer(x_passive)
        x_concat = tf.concat([x_active, x_passive], axis=1)
        return self.d4(x_concat)

    def provide_connect_gradient(self, gradients):
        # TODO: get correct logic of gradient when using eager execution
        if tf.executing_eagerly():
            return [gradi.numpy() for gradi in gradients[-2:]]
        res = []
        for gradi in gradients:
            if self.passive_connect_layer_name in gradi.name:
                res.append(tf.make_ndarray(gradi))
        return res

    def get_config(self):
        config = {}
        config.update({"d1_active": self.d1_active,
                       "d2_active": self.d2_active,
                       "d3_active": self.d3_active,
                       "d4": self.d4,
                       "passive_connect_layer": self.passive_connect_layer,
                       "passive_connect_layer_name": self.passive_connect_layer_name,
                      })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


#TODO: support @tf.function
def train_step(active_model, train_helper, connect_tensor):
    data, labels = train_helper.get_batch()
    with tf.GradientTape() as tape_active:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = active_model([data, connect_tensor], training=True)
        loss = train_helper.loss_object(labels, predictions)
        gradients_active = tape_active.gradient(loss, active_model.trainable_variables)
        connect_layer_gradients = active_model.provide_connect_gradient(gradients_active)
    train_helper.optimizer.apply_gradients(zip(gradients_active, active_model.trainable_variables))
    if train_helper.end_of_epoch:
        loss_report = train_helper.train_loss.result() * 100
        accuracy_report = train_helper.train_accuracy.result() * 100
        end_of_epoch = True
        train_helper.reset_epoch()
    else:
        loss_report = -1.
        accuracy_report = -1.
        end_of_epoch = False
    train_helper.train_loss(loss)
    train_helper.train_accuracy(labels, tf.squeeze(predictions))
    return {"connect_layer_gradients": connect_layer_gradients,
            "train_loss": loss_report,
            "train_accuracy": accuracy_report,
            "end_of_epoch": end_of_epoch}


def inference_step(model_active, train_helper, connect_tensor):
    data = train_helper.get_inference_data()
    output = model_active([data, connect_tensor])
    return str(output.numpy().tolist())


def get_or_assign_model_active(model_token,
                               passive_connect_layer_weights=None,
                               parameters=None,
                               data=None):
    global _DEEP_MODEL

    # get data
    # TODO: optimize logic
    if model_token == _DEEP_MODEL[0]:
        # model is being hold in _DEEP_MODEL
        model = _DEEP_MODEL[1]
        if data is None:
            train_helper = _DEEP_MODEL[2]
        else:
            if "y" in data:
                data = {"x": numpy.array(data["x"]).astype("float32"),
                        "y": numpy.array(data["y"]).astype("float32")}
                train_helper = TrainHelper.VFDNNTrainHelper(
                    data, parameters, is_active=True, training=True)
            else:
                data = {"x": numpy.array(data["x"]).astype("float32")}
                train_helper = TrainHelper.VFDNNTrainHelper(
                    data, parameters, is_active=True, training=False)
    elif os.path.exists(_CONFIG["base_path"] + model_token + "/"):
        # load model from path and create train helper
        _DEEP_MODEL[0] = model_token
        _DEEP_MODEL[1] = keras.models.load_model(_CONFIG["base_path"] + model_token + "/")
        if data is None:
            data = get_data_active()
            train_helper = TrainHelper.VFDNNTrainHelper(
                data, parameters, is_active=True, training=True)
        else:
            if "y" in data:
                data = {"x": numpy.array(data["x"]).astype("float32"),
                        "y": numpy.array(data["y"]).astype("float32")}
                train_helper = TrainHelper.VFDNNTrainHelper(
                    data, parameters, is_active=True, training=True)
            else:
                data = {"x": numpy.array(data["x"]).astype("float32")}
                train_helper = TrainHelper.VFDNNTrainHelper(
                    data, parameters, is_active=True, training=False)
    else:
        # create and train a new model
        # get model type
        model_type = model_token.split("_")[-1]
        if model_type == "MLP":
            model = ActiveMLPModel(passive_connect_layer_weights)
        else:
            raise ValueError("Invalid model type!")
        # create train helper
        if data is None:
            data = get_data_passive()
            train_helper = TrainHelper.VFDNNTrainHelper(
                data, parameters, is_active=True, training=True)
        else:
            data = {"x": numpy.array(data["x"]).astype("float32"),
                    "y": numpy.array(data["y"]).astype("float32")}
            train_helper = TrainHelper.VFDNNTrainHelper(
                data, parameters, is_active=True, training=True)
    _DEEP_MODEL = (model_token, model, train_helper)
    return model, train_helper


def get_data_active(data=None):
    if data is None:
        g = pandas.read_csv(_TRAIN_FILE)
        return {"x": g.loc[:, g.columns[1:-1]].values.astype("float32"),
                "y": g.loc[:, g.columns[-1]].values.astype("float32")}
    else:
        raise NotImplementedError("Not implement yet!")


def save_model(model_token, model):
    model.save(_CONFIG["base_path"] + model_token + "_active/")
    return None
