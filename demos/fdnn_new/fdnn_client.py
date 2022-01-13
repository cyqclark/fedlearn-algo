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

"""Example code of federated deep neural networks client.


Example
-------
TBA::

    $ TBA


Notes
-----
    This is an example of federated deep neural networks client. It assumes the
default folder is './opensource'.


"""
import argparse
import os
import sys

from importlib.machinery import SourceFileLoader

import numpy
import pandas

# assume cwd is ./opensource/
sys.path.append(os.getcwd())
from core.entity.common.machineinfo import MachineInfo
from core.entity.common.message import RequestMessage, ResponseMessage
from core.client.client import Client
from core.grpc_comm.grpc_server import serve

from active_model import ActiveModel
from passive_model import PassiveModel
from typing import Dict, List
from trainer import PassiveTrainer, ActiveTrainer
from table_data import TableData
import utils
import torch


class FDNNClient(Client):
    """
    FDNN client class.

    Note
    ----

    Parameters
    ----------
    machine_info : MachineInfo
        The machine info class that save the current client information,
        including ip, port and token.
    parameter : Dict
        The parameter of federated random forest model for this training task.
    dataset : Dict
        The binding dataset of this training task, including 'feature' and 'label' key.

    Attributes
    ----------
    machine_info : MachineInfo
        The machine_info argument passing in.
    parameter : Dict
        The parameter argument passing in.
    dataset : Dict
        The dataset argument passing in.
    """

    def __init__(self, machine_info: str, parameter: Dict):
        # super.__init__(parameter)
        # pass arguments
        self.parameter = parameter
        self.machine_info = machine_info
        self.training_dataset = None
        self.validation_dataset = None
        # set function mapping
        self.dict_functions = {"0": self.VFDNN_client_passive_initialization,
                               "1": self.VFDNN_client_active_initialization,
                               "2": self.VFDNN_client_passive_forward,
                               "3": self.VFDNN_client_active_forward_backward,
                               "4": self.VFDNN_client_passive_backward,
                               "finish": self.VFDNN_client_post_training,
                               "-1": self.VFDNN_client_inference_phase1,
                               "-2": self.VFDNN_client_inference_phase2}
        # no preprocessing or postprocessing in this demo training code
        self.preprocessing_func = {}
        self.model = None
        self.trainer = None
        self.is_active = False
        self.hyper_parameters = utils.HyperParameters()
        # check if use remote mode
      
    def load_training_data(self, path: str, feature_names: List, label: str= None):
        self.training_dataset = TableData()
        self.training_dataset.load_data(path=path, feature_names=feature_names, label=label)

    def load_validation_data(self, path: str, feature_names: List, label: str= None):
        self.validation_dataset = TableData()
        self.validation_dataset.load_data(path=path, feature_names=feature_names, label=label)

    def make_response(self, request, body):
        """
        Making response function. Given the input request and body dictionary,
        this function returns a response object which is ready for sending out.

        Parameters
        ----------
        request : RequestMessage
            The request message sending into the client.
        body : Dict
            The response body data in dictionary type.

        Returns
        -------
        ResponseMessage
            The response message which is ready for sending out.
        """
        # check if the current client is an active client.

        response = ResponseMessage(self.machine_info,
                                   request.server_info,
                                   body,
                                   phase_id=request.phase_id)
        response.serialize_body()
        return response

    def control_flow_client(self, phase_num,request):
        """
        The main control flow of client. This might be able to work in a generic
        environment.
        """
        # if phase has preprocessing, then call preprocessing func
        response = request
        if phase_num in self.preprocessing_func:
            response = self.preprocessing_func[phase_num](response)
        if phase_num in self.dict_functions:
            response = self.dict_functions[phase_num](response)
        # if phase has postprocessing, then call postprocessing func
        #if phase_num in self.postprocessing_func:
        #    response = self.postprocessing_func[phase_num](response)
        return response

    def initialization_client(self, request):
        """
        Client initialization function.

        Parameters
        ----------
        request : RequestMessage
            The request message sending into the client.

        Returns
        -------
        """
        self.is_active = self.training_dataset.data['label'] is not None
        if self.training_dataset.data['features'] is not None:
            self.training_dataset.data_preprocessing()
            self.training_dataset.data_transform()

    def check_ser_deser(self, message):
        if isinstance(message, RequestMessage):
            message.deserialize_body()
        elif isinstance(message, ResponseMessage):
            message.serialize_body()
        return None

    def VFDNN_client_passive_initialization(self, request):
        """
        Passive parties initialize the model training.
        """
        self.initialization_client(request)

        # skip active party
        if self.is_active:
            body = {'is_active': self.is_active, 'token': self.machine_info.token}
            return self.make_response(request, body)

        self.model = PassiveModel(self.training_dataset)
        body = {'remote_input_dim': self.model.out_dim, 'is_active': self.is_active, 'token': self.machine_info.token}
        return self.make_response(request, body)

    def VFDNN_client_active_initialization(self, request):
        self.check_ser_deser(request)
        # skip passive party
        if not self.is_active:
            return self.make_response(request, {'is_active': self.is_active})

        # parse passive model weights
        
        remote_input_dim = request.body['remote_input_dim']
        self.model = ActiveModel(self.training_dataset, remote_input_dim)
        self.hyper_parameters.batch_size = 500
        body = {'is_active': self.is_active, 'batch_size': self.hyper_parameters.batch_size,
                'sample_number': self.training_dataset.sample_number} #self.make_body(d["model_token"], [], d["model_parameters"])

        return self.make_response(request, body)

    def VFDNN_client_passive_forward(self, request):
        """
        Phase2: Passive model generate connect layer features
        """
        self.check_ser_deser(request)
        # skip active party
        if self.is_active:
            body = {'is_active': self.is_active}
            if 'sample_index' in request.body:
                sample_index = request.body['sample_index']
                self.training_dataset.data_shuffle(sample_index)
            return self.make_response(request, body)
        if 'sample_index' in request.body:
            sample_index = request.body['sample_index']
            self.training_dataset.data_shuffle(sample_index)
            
        self.hyper_parameters.batch_size = request.body['batch_size']
        batch_index = request.body['batch_index']
        self.trainer = PassiveTrainer(self.model, self.hyper_parameters)
        self.trainer.batch_size = request.body['batch_size']
        self.trainer.sample_number = self.training_dataset.sample_number
        mini_batch = self.trainer.generate_mini_batch(self.training_dataset, batch_index)
        self.trainer.model_forward(mini_batch)
        output = self.trainer.get_output()

        body = {self.machine_info.token: output, 'is_active': self.is_active}
        return self.make_response(request, body)

    def VFDNN_client_active_forward_backward(self, request):
        """
        Phase3: Train active model
        """
        self.check_ser_deser(request)
        # skip passive party
        if not self.is_active:
            body = {'is_active': self.is_active}
            return self.make_response(request, body)

        if 'sample_index' in request.body:
            sample_index = request.body['sample_index']
            self.training_dataset.shuffle(sample_index)

        remote_input = request.body['remote_inputs']
        batch_index = request.body['batch_index']
        
        self.trainer = ActiveTrainer(self.model, self.hyper_parameters)
        self.trainer.batch_size = request.body['batch_size']
        self.trainer.sample_number = self.training_dataset.sample_number
        mini_batch = self.trainer.generate_mini_batch(self.training_dataset, batch_index)
        out = self.trainer.model_forward(mini_batch, remote_input)
        self.model = self.trainer.model_backward(out, mini_batch.label)
        gradients = self.trainer.get_remote_gradients()

        body = {'is_active': self.is_active, 'gradients': gradients}
        return self.make_response(request, body)

    def VFDNN_client_passive_backward(self, request):
        """
        Phase4: Train passive model
        """
        self.check_ser_deser(request)
        # skip active party
        if self.is_active:
            body = {'is_active': self.is_active, 'batch_size': 2,
                    'sample_number': self.training_dataset.sample_number}
            return self.make_response(request, body)

        gradient = request.body['gradient']
        self.trainer.model_backward(gradient)
        body = {'is_active': self.is_active}
        return self.make_response(request, body)

    def VFDNN_client_post_training(self, request):
        # TODO: serialization and save
        path = 'model_'+self.machine_info.token
        torch.save({'model':self.model, 'num_feat_scaler': self.training_dataset.num_scalers,
                    'cate_feature_encoder':self.training_dataset.cate_encoders}, path)
        return self.make_response(request, {})

    def VFDNN_client_inference_phase1(self, request):
        self.check_ser_deser(request)
        d = request.body
        model_token = d["model_token"]
        data = {"x": self.load_inference_data()}
        model, train_helper = PassiveModel.get_or_assign_model_passive(
            model_token, data=data)
        feature = PassiveModel.get_connect_layer_feature(
            model, train_helper, training=False)
        model_bytes = util.serialize_tf_tensor(feature, method="numpy")
        body = self.make_body(d["model_token"], [model_bytes], [])
        return self.make_response(request, body)

    def VFDNN_client_inference_phase2(self, request):
        self.check_ser_deser(request)
        d = request.body
        model_token = d["model_token"]
        passive_feature = d["model_bytes"][0]
        passive_feature = util.deserialize_tf_tensor(passive_feature, method="numpy")
        data = {"x": self.load_inference_data()}
        model, train_helper = ActiveModel.get_or_assign_model_active(
            model_token, data=data)
        res = ActiveModel.inference_step(model, train_helper, passive_feature)
        model_bytes = [res.encode("utf-8")]
        body = self.make_body(model_token, model_bytes, {})
        return self.make_response(request, body)

    def load_inference_data(self):
        return self.dataset["feature_inference"]


if __name__ == "__main__":
    # for single client
    parser = argparse.ArgumentParser()
    parser.add_argument('-I', '--index', type=int, required=True, help='index of client')
    parser.add_argument('-C', '--python_config_file', type=str, required=True, help='python config file')

    args = parser.parse_args()
    idx = args.index
    config = SourceFileLoader("config", args.python_config_file).load_module()

    # load data
    g = pandas.read_csv(config.client_train_file_path[idx])
    if idx == config.active_index:
        label = g.pop(config.active_label).values.ravel().astype(float)
        dataset = {"label": label,
                   "feature": g.loc[:, g.columns[1:]].values}
    else:
        dataset = {"label": None,
                   "feature": g.loc[:, g.columns[1:]].values}
    if "client_inference_file_path" in config.__dict__:
        g = pandas.read_csv(config.client_inference_file_path[idx])
        dataset["feature_inference"] = g.loc[:, g.columns[1:]].values
    ip, port = config.client_ip_and_port[idx].split(":")
    client_info = MachineInfo(ip=ip, port=port,
                              token=config.client_ip_and_port[idx])

    parameter = config.parameter
    client = FDNNClient(client_info, parameter, dataset, remote=True)

    serve(client)

