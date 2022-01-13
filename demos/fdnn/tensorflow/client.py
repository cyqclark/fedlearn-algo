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

import ActiveModel, PassiveModel, util
from coordinator import FDNNCoordinator


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

    def __init__(self,
                 machine_info,
                 parameter,
                 dataset,
                 remote=False):
        # super.__init__(parameter)
        # pass arguments
        super().__init__(machine_info)
        self.parameter = parameter
        self.machine_info = machine_info
        self.dataset = dataset
        # set function mapping
        self.dict_functions = {"0": self.VFDNN_client_phase0_passive,
                               "1": self.VFDNN_client_phase0_active,
                               "2": self.VFDNN_client_phase2,
                               "3": self.VFDNN_client_phase3,
                               "4": self.VFDNN_client_phase4,
                               "99": self.VFDNN_client_post_training,
                               "-1": self.VFDNN_client_inference_phase1,
                               "-2": self.VFDNN_client_inference_phase2,
                               }
        # no preprocessing or postprocessing in this demo training code
        self.preprocessing_func = {}
        self.postprocessing_func = {}

        # check if use remote mode
        self.remote = remote
        return None

    def make_body(self, model_token, model_bytes, model_parameters):
        return {"model_token": model_token,
                "model_bytes": model_bytes,
                "model_parameters": model_parameters}

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
        if "is_active" not in body:
            body["is_active"] = self.is_active
        if "model_token" not in body:
            body["model_token"] = request.body["model_token"]
        if "model_parameters" not in body:
            body["model_parameters"] = request.body["model_parameters"]
        response = ResponseMessage(self.machine_info,
                                   request.server_info,
                                   body,
                                   phase_id=request.phase_id)
        if self.remote:
            response.serialize_body()
        return response

    def make_prediction(self, treeid, nodeid):
        """
        Make prediction function.
        """
        sample_id = self.forest[treeid]["tree"][nodeid]["sample_id"]
        self.forest[treeid]["tree"][nodeid]["is_leaf"] = True
        self.forest[treeid]["tree"][nodeid]["prediction"] = numpy.mean(self.dataset["label"][sample_id])
        return None

    def update_split_info(self, treeid, feature, value):
        """
        Updating split information function.
        """
        nodeid = self.current_nodes[treeid]
        self.forest[treeid]["tree"][nodeid]["feature"] = feature
        self.forest[treeid]["tree"][nodeid]["value"] = value
        self.forest[treeid]["tree"][nodeid]["is_leaf"] = False
        return None

    def control_flow_client(self,
                            phase_num,
                            request):
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
        if phase_num in self.postprocessing_func:
            response = self.postprocessing_func[phase_num](response)
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
        self.is_active = self.dataset["label"] is not None
        self.label = self.dataset["label"]
        body = {}
        return self.make_response(request, body)

    def check_ser_deser(self, message):
        if self.remote:
            if isinstance(message, RequestMessage):
                message.deserialize_body()
            elif isinstance(message, ResponseMessage):
                message.serialize_body()
        return None

    def VFDNN_client_phase0_passive(self, request):
        self.check_ser_deser(request)
        self.initialization_client(request)

        # skip active party
        if self.is_active:
            return self.make_response(request, {})

        d = request.body
        model_token = d["model_token"]
        data = {"x": self.dataset["feature"]}
        parameters = self.parameter
        model, _ = PassiveModel.get_or_assign_model_passive(
            model_token, parameters, data)
        connect_layer_weights = model.get_connect_layer_weights()
        model_bytes = []
        for weighti in connect_layer_weights:
            bytei = util.serialize_numpy_array(weighti, method="numpy")
            model_bytes.append(bytei)
        body = self.make_body(model_token, model_bytes, d["model_parameters"])
        return self.make_response(request, body)

    def VFDNN_client_phase0_active(self, request):
        self.check_ser_deser(request)
        # skip passive party
        if not self.is_active:
            return self.make_response(request, {})

        d = request.body
        model_token = d["model_token"]
        parameters = self.parameter
        # parse passive model weights
        tmp = d["model_bytes"]
        weights = []
        for bytei in tmp:
            weighti = util.deserialize_numpy_array(bytei, method="numpy")
            weights.append(weighti)
        data = {}
        data["x"] = self.dataset["feature"]
        data["y"] = self.dataset["label"]
        model, trainer = ActiveModel.get_or_assign_model_active(
            model_token,
            weights,
            parameters,
            data)
        body = self.make_body(d["model_token"], [], d["model_parameters"])
        return self.make_response(request, body)

    def VFDNN_client_phase2(self, request):
        """
        Phase2: Passive model generate connect layer feature
        """
        self.check_ser_deser(request)
        # skip active party
        if self.is_active:
            return self.make_response(request, request.body)
        d = request.body
        model_token = d["model_token"]
        model, trainer = PassiveModel.get_or_assign_model_passive(model_token)
        feature = PassiveModel.get_connect_layer_feature(model, trainer)
        model_bytes = util.serialize_tf_tensor(feature, method="numpy")
        body = self.make_body(d["model_token"], [model_bytes], [])
        return self.make_response(request, body)

    def VFDNN_client_phase3(self, request):
        """
        Phase3: Train active model
        """
        self.check_ser_deser(request)
        # skip passive party
        if not self.is_active:
            return self.make_response(request, request.body)
        d = request.body
        model_token = d["model_token"]
        passive_feature = d["model_bytes"][0]
        passive_feature = util.deserialize_tf_tensor(passive_feature, method="numpy")
        model, trainer = ActiveModel.get_or_assign_model_active(model_token)
        res = ActiveModel.train_step(model, trainer, passive_feature)
        if res["end_of_epoch"]:
            # end of epoch update loss
            parameters = [1, res["train_loss"], res["train_accuracy"]]
        else:
            parameters = [0]
        gradients = res["connect_layer_gradients"]
        model_bytes = []
        for gradi in gradients:
            bytei = util.serialize_numpy_array(gradi, method="numpy")
            model_bytes.append(bytei)
        body = self.make_body(d["model_token"], model_bytes, parameters)
        return self.make_response(request, body)

    def VFDNN_client_phase4(self, request):
        """
        Phase4: Train passive model
        """
        self.check_ser_deser(request)
        # skip active party
        if self.is_active:
            return self.make_response(request, request.body)
        d = request.body
        model_token = d["model_token"]
        byte_gradients = d["model_bytes"]
        gradients = []
        for bytei in byte_gradients:
            gradi = util.deserialize_numpy_array(bytei, method="numpy")
            gradients.append(gradi)
        model, trainer = PassiveModel.get_or_assign_model_passive(model_token)
        PassiveModel.train_step_passive(model, trainer, gradients)
        body = self.make_body(d["model_token"], [], [])
        return self.make_response(request, body)

    def VFDNN_client_post_training(self, request):
        # TODO: serialization and save
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
    parser.add_argument('-F', '--flag_network', type=str, required=False, default="F", help='flag to use new network api')

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

    if ("flag_network" not in args) or (args.flag_network == "F"):
        # old api framework
        serve(client)
    elif args.flag_network == "T":    
        # set active client
        if config.active_index == args.index:
            coordinator_info = MachineInfo(ip=ip, port=port,
                                           token=config.coordinator_ip_and_port)
            client_infos = []
            for ci in config.client_ip_and_port:
                ip, port = ci.split(":")
                client_infos.append(MachineInfo(ip=ip, port=port, token=ci))
            coordinator = FDNNCoordinator(coordinator_info,
                                          client_infos,
                                          config.parameter,
                                          remote=True)
            client.load_coordinator(coordinator)
            client._exp_training_pipeline("0")
        else:
            serve(client)
    else:
        raise ValueError("Invalid flag network")


