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

"""Example code of FDNN algorithm coordinator.


Example
-------
TBA::

    $ TBA


Notes
-----
    This is an example of FDNN algorithm coordinator. It assumes the
default folder is './opensource'.


"""
import argparse
import copy
import os
import sys

from importlib.machinery import SourceFileLoader

import orjson

import numpy

# assume cwd is ./opensource/
sys.path.append(os.getcwd())
from core.entity.common.machineinfo import MachineInfo
from core.entity.common.message import RequestMessage, ResponseMessage
from core.server.server import Server
import core.server.server


class FDNNServer(Server):
    """
    FDNN server class.

    Note
    ----
    Parameters
    ----------
    server_info: machineInfo
        The machine info class that saving the current coordinator information,
        including ip, port and token.
    clients_info : List of machineInfo
        The list of that saving the machine info of clients, including ip,
        port and token.
    parameter : Dict
        The parameter of FDNN algorithm model for this training task.
    """

    def __init__(self,
                 server_info,
                 clients_info,
                 parameter):
        super().__init__()
        # pass arguments
        self.parameter = parameter
        self.server_info = server_info
        self.clients_info = clients_info
        # set function mapping
        self.dict_functions = {"1": self.fdnn_coordinator_phase1,
                               "2": self.fdnn_coordinator_phase2,
                               "3": self.fdnn_coordinator_phase3,
                               "4": self.fdnn_coordinator_phase4,
                               "-1": self.fdnn_coordinator_inference_phase1,
                               "-2": self.fdnn_coordinator_inference_phase2,
                               "-3": self.fdnn_coordinator_inference_phase3}
        # no preprocessing or postprocessing in this demo training code
        self.preprocessing_func = {}
        self.postprocessing_func = {}

        self.train_finish = False
        self.inference_finish = False

        # VFDNN parameter
        self.current_epoch = 0

        self.sample_index = None
        self.sample_num = 0
        
        self.batch_size = 0
        self.batch_index = 1
        self.batch_num = None
        
        self.epoch_num = 30
        self.epoch_index = 0

    def is_training_continue(self):
        """
        Check if training is finished. Inherit from super class.
        """
        if self.epoch_index < self.epoch_num:
            return True
        else:
            return False

    def is_inference_continue(self):
        """
        Check if inference is finished. Inherit from super class.
        """
        return not self.inference_finish

    def check_if_finish(self):
        """
        Check if training is finished.
        """
        self.train_finish = self.current_epoch >= self.parameter["num_epoch"]

    def control_flow_coordinator(self,
                                 phase_num,
                                 responses):
        """
        The main control flow of coordinator. This might be able to work in a generic
        environment.
        """
        # update phase id
        for res in responses:
            res.phase_id = phase_num
        # if phase has preprocessing, then call preprocessing func
        if phase_num in self.dict_functions:
            requests = self.dict_functions[phase_num](responses)
        else:
            raise ValueError("Cannot find correct function with the input phase_num")
        return requests

    def get_next_phase(self, old_phase):
        """
        Given old phase, return next phase.
        The logic is:
            0 => 1 => 2 => 3 => 4 => 2 => 3 => 4 => 2 ...
        """
        if int(old_phase) >= 0:
            if old_phase == "0":
                next_phase = "1"
            elif (old_phase == "1") or (old_phase == "4"):
                next_phase = "2"
            elif old_phase == "2":
                next_phase = "3"
            elif old_phase == "3":
                next_phase = "4"
            else:
                raise ValueError("Invalid phase number")
        elif int(old_phase) < 0:
            if old_phase == "-1":
                next_phase = "-2"
            elif old_phase == "-2":
                next_phase = "-3"
            else:
                raise ValueError("Invalid phase number")
        else:
            raise ValueError("Invalid phase number")
        return next_phase

    def check_ser_deser(self, message):
        if isinstance(message, ResponseMessage):
            message.deserialize_body()
        elif isinstance(message, RequestMessage):
            message.serialize_body()
        return None

    def init_training_control(self):
        """
        Create initial requests.
        """
        # control init passive
        body = {"model_token": "123_MLP",
                "model_parameters": self.parameter}
        requests = {client: RequestMessage(self.server_info, client, body, "0")
                    for client in self.clients_info}
        for _, req in requests.items():
            self.check_ser_deser(req)
        return requests

    def make_request(self, response, body, phase_id):
        """
        Making request function. Given the input response and body dictionary,
        this function returns a request object which is ready for sending out.

        Parameters
        ----------
        response : ResponseMessage
            The response message sending into the client.
        body : Dict
            The request body data in dictionary type.

        Returns
        ------- 
        RequestMessage
            The request message which is ready for sending out.
        """
        request = RequestMessage(self.server_info,
                                 response.client_info,
                                 body,
                                 phase_id)
        self.check_ser_deser(request)
        return request

    def fdnn_coordinator_phase1(self, responses):
        """
        Coordinator phase 1 code of FDNN algorithm.

        Parameters
        ----------
        responses : List
            The list of ResponseMeesage sending back from the clients.

        Returns
        ------- 
        List
            The list of RequestMessage sending to the clients.
        """
        # control init active
        requests = {}

        feat_dim = {}
        total_dim = 0
        for client_info, res in responses.items():
            self.check_ser_deser(res)
            if not res.body["is_active"]:
                total_dim += res.body['remote_input_dim']
                token = res.body['token']
                feat_dim[token] = res.body['remote_input_dim']
        body = {'remote_input_dim': feat_dim}
        for client_info, res in responses.items():
            requests[client_info] = self.make_request(res, body, "1")

        return requests

    def fdnn_coordinator_phase2(self, responses):
        """
        Coordinator phase 2 code of FDNN algorithm.

        Parameters
        ----------
        responses : List
            The list of ResponseMessage sending back from the clients.

        Returns
        ------- 
        List
            The list of RequestMessage sending to the clients.
        """
        # control init active
        requests = {}
        body = {}
        for client_info, res in responses.items():
            self.check_ser_deser(res)
            if res.body['is_active']:
                self.sample_num = res.body['sample_number']
                self.batch_size = res.body['batch_size']
                self.batch_num = numpy.ceil(self.sample_num/self.batch_size)
                body = {'batch_size': self.batch_size, 'batch_index': self.batch_index,
                        'batch_num': self.batch_num}
            if self.batch_index > self.batch_num:
                self.batch_index = 1
                self.epoch_index += 1
                self.sample_index = [i for i in range(self.sample_num)]
                numpy.random.shuffle(self.sample_index)
                body = {'batch_size': self.batch_size, 'batch_index': self.batch_index,
                        'batch_num': self.batch_num, 'sample_index': self.sample_index}
        for client_info, res in responses.items():
            requests[client_info] = self.make_request(res, copy.deepcopy(body), "2")

        return requests

    def fdnn_coordinator_phase3(self, responses):
        """
        Coordinator phase 3 code of FDNN algorithm.
        Active party collect all
        Parameters
        ----------
        responses : List
            The list of ResponseMessage sending back from the clients.

        Returns
        ------- 
        List
            The list of RequestMessage sending to the clients.
        """
        requests = {}
        remote_inputs = {}
        # collect model bytes
        for client_info, res in responses.items():
            self.check_ser_deser(res)
            if not res.body["is_active"]:
                token = client_info.token
                remote_inputs[token] = res.body[token]
        for client_info, res in responses.items():
            if res.body["is_active"]:
                body = {"remote_inputs" : remote_inputs, 'batch_index': self.batch_index, 'batch_size': self.batch_size}
            requests[client_info] = self.make_request(res, copy.deepcopy(body), "3")
        self.batch_index += 1
        return requests

    def fdnn_coordinator_phase4(self, responses):
        """
        Coordinator phase 4 code of FDNN algorithm.

        Parameters
        ----------
        responses : List
            The list of ResponseMessage sending back from the clients.

        Returns
        ------- 
        List
            The list of RequestMessage sending to the clients.
        """
        requests = {}
        gradients = []

        for client_info, res in responses.items():
            self.check_ser_deser(res)
            body = res.body
            if body["is_active"]:
                gradients = body["gradients"]

        for client_info, res in responses.items():
            body = {'batch_size': self.batch_size}
            if not res.body["is_active"]:
                body["gradient"] = gradients[client_info.token]
            else:
                body['is_active'] = False
            requests[client_info] = self.make_request(res, copy.deepcopy(body), "4")
        # check finish
        # self.check_if_finish()
        return requests

    def fdnn_control_flow_finish(self):
        requests = {}
        for client_info in self.clients_info:
            requests[client_info] = RequestMessage(sender=self.server_info, receiver=client_info, body={}, phase_id='finish')
        return requests

    def fdnn_coordinator_inference_phase1(self, responses):
        """
        Coordinator inference phase 1 code of FDNN algorithm.

        Parameters
        ----------
        responses : List
            The list of ResponseMessage sending back from the clients.

        Returns
        ------- 
        List
            The list of RequestMessage sending to the clients.
        """
        requests = {}

        for client_info, resi in responses.items():
            self.check_ser_deser(resi)
            body = resi.body
            requests[client_info] = self.make_request(resi, copy.deepcopy(body), "-1")
        return requests

    def fdnn_coordinator_inference_phase2(self, responses):
        """
        Coordinator inference phase 2 code of FDNN algorithm.

        Parameters
        ----------
        responses : List
            The list of ResponseMessage sending back from the clients.

        Returns
        ------- 
        List
            The list of RequestMessage sending to the clients.
        """
        requests = {}
        passive_body = []
        body = {}

        for client_info, resi in responses.items():
            self.check_ser_deser(resi)
            if not resi.body["is_active"]:
                passive_body.append(resi.body)
        for client_info, resi in responses.items():
            if resi.body["is_active"]:
                body["model_token"] = passive_body[0]["model_token"]
                body["model_bytes"] = passive_body[0]["model_bytes"]
                body["is_active"] = resi.body["is_active"]
                requests[client_info] = self.make_request(resi, body, "-2")
            else:
                requests[client_info] = self.make_request(resi, resi.body, "-2")

        return requests

    def fdnn_coordinator_inference_phase3(self, responses):
        """
        Coordinator inference phase 3 code of FDNN algorithm.

        Parameters
        ----------
        responses : List
            The list of ResponseMessage sending back from the clients.

        Returns
        ------- 
        List
            The list of RequestMessage sending to the clients.
        """

        for client_info, resi in responses.items():
            self.check_ser_deser(resi)
            if resi.body["is_active"]:
                self.prediction = orjson.loads(resi.body["model_bytes"][0])
        self.inference_finish = True
        return self.make_null_requests(responses, "3")

    # abs
    def post_inference_session(self):
        return self.prediction

    def post_training_session(self):
        body = {"model_token": "",
                "model_parameters": []}
        requests = {clienti: RequestMessage(self.machine_info_coordinator, clienti, body, "99")
                    for clienti in self.machine_info_client}
        return requests

    def init_inference_control(self):
        """
        Init inference
        """
        body = {"model_token": "123_MLP",
                }
        requests = {client_info: RequestMessage(self.machine_info_coordinator, client_info, body, "-1")
                    for client_info in self.machine_info_client}
        self.prediction = None
        return requests

    def metric(self):
        return None


if __name__ == "__main__":
    # for single coordinator
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--python_config_file', type=str, required=True, help='python config file')

    args = parser.parse_args()
    config = SourceFileLoader("config", args.python_config_file).load_module()

    ip, port = config.coordinator_ip_and_port.split(":")
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

    # train
    coordinator.training_pipeline("0", client_infos, is_parallel=False)
    # inference