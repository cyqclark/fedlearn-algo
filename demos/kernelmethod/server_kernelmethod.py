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

"""This file is the server implementation of kernel method."""

from core.entity.common.message import RequestMessage, ResponseMessage
from core.entity.common.machineinfo import MachineInfo
from core.server.server import Server

import numpy
from typing import Dict, Set


class KernelMethodServerError(ValueError):
    pass


def check_message(msg: str, target: Set):
    if msg in target:
        return True
    else:
        return False


class KernelMethodsServer(Server):
    def __init__(self, machine_info: MachineInfo):
        super().__init__()
        self.machine_info = machine_info
        self.clients_info = None
        self.max_iter = 1000
        self.iter = 0
        self.sample_num = 0
        self.clients_token = []
        self.prediction = []
        self.dict_functions = {}
        self.inference_communication_count = 0
        self.function_registration()
        self.machine_ind = 0

    def metric(self) -> None:
        return

    def get_next_phase(self, phase: str) -> str:
        """
        Transfer old phase of client to new phase of server
        """
        # train
        if phase == "train_init":
            next_phase = "train_loop_start"
        elif (phase == "train_loop_start") or (phase == "param_update"):
            next_phase = "meta_comp"
        elif phase == "meta_comp":
            next_phase = "param_update"
        # inference
        elif phase == "inference_init":
            next_phase = "inference_comp"
        elif phase == "inference_comp":
            next_phase = "inference_end"
        # raise error
        else:
            raise ValueError("Cannot find phase %s in both train and inference!"%phase)
        return next_phase

    def function_registration(self):
        """
        Define the correspondence between response message type and server processing function.
        """
        # train
        self.dict_functions["train_loop_start"] = self.train_loop_start
        self.dict_functions["meta_comp"] = self.req_tr_meta_comp
        self.dict_functions["param_update"] = self.param_update
        # inference
        self.dict_functions["inference_comp"] = self.req_infer_comp
        self.dict_functions["inference_end"] = self.predict

    # Training related code implementation.

    def init_training_control(self) -> Dict[MachineInfo, RequestMessage]:
        """
        Send training initialization request to clients.
        """
        requests = {}
        for client_info in self.clients_info:
            requests[client_info] = RequestMessage(sender=self.machine_info,
                                                   receiver=client_info,
                                                   body=None,
                                                   phase_id="train_init")
        return requests

    def is_training_continue(self) -> bool:
        if self.iter <= self.max_iter:
            return True
        else:
            return False

    def is_inference_continue(self) -> bool:
        return self.inference_communication_count < 3

    def train_loop_start(self, responses: Dict[MachineInfo, ResponseMessage]) -> Dict[MachineInfo, RequestMessage]:
        """
        Training loop starts.
        """
        requests = {}
        for client_info, response in responses.items():
            self.sample_num = response.body['sample_num']
            request = RequestMessage(sender=self.machine_info,
                                     receiver=client_info,
                                     body=None,
                                     phase_id="train_loop_start")
            requests[client_info] = request
        return requests

    def param_update(self, responses: Dict[MachineInfo, ResponseMessage]) -> Dict[MachineInfo, RequestMessage]:
        """
        Aggregate the update from clients, then send request to clients for local parameter update.
        The input response message contains all clients' meta computation update.
        """
        vec_sum = numpy.zeros((self.sample_num, 1), dtype=numpy.float)
        for client_info, response in responses.items():
            body = response.body
            vec_sum += body['meta_result']

        loss = numpy.dot(vec_sum.T, vec_sum)/self.sample_num
        print('training loss at iteration ' + str(self.iter) + ' is ' +str(loss))

        client_num = len(self.clients_token)
        self.machine_ind += 1
        if self.machine_ind >= client_num:
            self.machine_ind = 0

        requests = {}
        body = {'aggregation_result': vec_sum, 'chosen_machine': self.clients_token[self.machine_ind]}
        for client_info, response in responses.items():
            request = RequestMessage(sender=self.machine_info,
                                     receiver=client_info,
                                     body=body,
                                     phase_id="param_update")
            requests[client_info] = request
        self.iter += 1
        return requests

    def req_tr_meta_comp(self, responses: Dict[MachineInfo, ResponseMessage]) -> Dict[MachineInfo, RequestMessage]:
        """
        Send request to clients to compute the meta results.
        """
        requests = {}
        for client_info, response in responses.items():
            request = RequestMessage(sender=self.machine_info,
                                     receiver=client_info,
                                     body=None,
                                     phase_id="meta_comp")
            requests[client_info] = request
        return requests

    def post_training_session(self) -> Dict[MachineInfo, RequestMessage]:
        """
        Send finish signal to clients.
        """
        requests = {}
        for client_info in self.clients_info:
            body = {'message': 'finish_training'}
            requests[client_info] = RequestMessage(sender=self.machine_info,
                                                   receiver=client_info,
                                                   body=body,
                                                   phase_id="train_finish")
        return requests

    # Inference related function code.

    def init_inference_control(self) -> Dict[MachineInfo, RequestMessage]:
        """
        Send request to clients for inference initialization.
        """
        requests = {}
        for client_info in self.clients_info:
            requests[client_info] = RequestMessage(sender=self.machine_info,
                                                   receiver=client_info,
                                                   body=None,
                                                   phase_id="inference_init")
        self.inference_communication_count += 1
        return requests

    def req_infer_comp(self, responses: Dict[MachineInfo, ResponseMessage]) -> Dict[MachineInfo, RequestMessage]:
        """
        Send request to clients for meta result compute.
        """
        requests = {}
        for client_info, response in responses.items():
            if check_message(response.body['message'], {'initialization_ready'}):
                self.sample_num = response.body['sample_num']
            else:
                raise KernelMethodServerError('inference on client %s fails to be initialized', client_info.token)
            request = RequestMessage(sender=self.machine_info,
                                     receiver=client_info,
                                     body=None,
                                     phase_id="inference_comp")
            requests[client_info] = request
        self.inference_communication_count += 1
        return requests

    def predict(self, responses: Dict[MachineInfo, ResponseMessage]) -> Dict[MachineInfo, RequestMessage]:
        """
        Finish the inference session.
        """
        requests = {}
        self.prediction = numpy.zeros((self.sample_num, 1), dtype=numpy.float)
        for client_info, response in responses.items():
            self.prediction += response.body['inner_product']
            request = RequestMessage(sender=self.machine_info,
                                     receiver=client_info,
                                     body=None,
                                     phase_id=None)
            requests[client_info] = request
        self.inference_communication_count += 1
        return requests

    def post_inference_session(self) -> None:
        print("Predictions: ")
        print(self.prediction)
        return None