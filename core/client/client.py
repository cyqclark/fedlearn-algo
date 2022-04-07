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

# This file is the class template theABC JDT client
from core.entity.common.message import RequestMessage, ResponseMessage
from core.grpc_comm.grpc_node import send_request
from core.grpc_comm.grpc_converter import grpc_msg_to_common_msg, common_msg_to_grpc_msg
from core.grpc_comm.grpc_node import GRPCNode
from core.grpc_comm.grpc_servicer import GRPCServicer
from core.proto.transmission_pb2 import ReqResMessage
from core.proto.transmission_pb2_grpc import TransmissionServicer

from abc import abstractmethod
from typing import Dict
import pickle

class ClientError(ValueError):
    pass


class Client(TransmissionServicer):
    """
    Basic client class
    """
    def __init__(self, client_info=None):
        self._client_info = client_info

    def __init__(self):
        self.grpc_servicer = GRPCServicer()
        self.grpc_node = GRPCNode()

    @property
    def dict_functions(self):
        """
        Dictionary of functions that store the training function mapping as
        <phase_id: training_function>.
        """
        return self._dict_functions
    
    @dict_functions.setter
    def dict_functions(self, value):
        if not isinstance(value, dict):
            raise ValueError("Funcion mapping must be a dictionary!")
        self._dict_functions = value

    @abstractmethod
    def train_init(self) -> None:
        """
        Training initialization function
        
        Returns 
        -------
        None
        """
        
    @abstractmethod
    def inference_init(self) -> None:
        """
        Inference initialization function
        
        Returns
        -------
        None
        """

    def load_model(self, model_path: str) -> Dict:
        """

        Parameters
        ----------
        model_path: str

        Returns
        -------
        model: dict
        
        """
        f = open(model_path, 'rb')
        model = pickle.load(f)
        f.close()
        return model

    def save_model(self, model_path: str, model: Dict) -> None:
        """

        Parameters
        ----------
        model_path: str

        model: dict

        Returns
        -------
        None
        """
        f = open(model_path, 'wb')
        pickle.dump(model, f)
        f.close()

    def process_request(self, request: RequestMessage) -> ResponseMessage:
        """

        Parameters
        ----------
        request: RequestMessage

        Returns
        -------
        response: ResponseMessage
        """
        symbol = request.phase_id
        if symbol not in self.dict_functions.keys():
            raise ClientError("Function %s is not implemented.", symbol)
        response = self.dict_functions[symbol](request)
        return response

    def comm(self, grpc_request: ReqResMessage, context) -> ReqResMessage:
        common_req_msg = grpc_msg_to_common_msg(grpc_request)
        common_res_msg = self.process_request(common_req_msg)
        return common_msg_to_grpc_msg(common_res_msg)

    # optional for coordinator
    def load_coordinator(self, coordinator):
        self.coordinator = coordinator
        self._has_coordinator = True
        return None

    def _exp_call_local_client(self, request):
        common_res_msg = self.process_request(request)
        return common_res_msg

    def _exp_send_grpc_request(self,
                                request):
        """
        Experiment function for new sending grpc request
        """
        client_info = request.client_info
        if (client_info.ip == self._client_info.ip) and (
            client_info.port == self._client_info.port):
            response = self._exp_call_local_client(request)
        else:
            response = send_request(request)
        return response

    def _exp_call_grpc_client(self, requests, is_parallel=True):
        responses = {}
        """
        if is_parallel:
            req_client_info_list = []
            parallel_grpc_client_thread_list = []
            for client_info, request in requests.items():
                req_client_info_list.append(client_info)
                temp_t = SendingThread(send_grpc_request, args=request)
                parallel_grpc_client_thread_list.append(temp_t)
            for i, t in enumerate(parallel_grpc_client_thread_list):
                t.setDaemon(True)
                t.start()
            for client_info, t in zip(req_client_info_list, parallel_grpc_client_thread_list):
                t.join()
                responses[client_info] = t.get_result()
        else:
        """
        for client_info, request in requests.items():
            responses[client_info] = self._exp_send_grpc_request(request)
        return responses

    def _exp_training_pipeline(self, init_phase: str, is_parallel=False) -> None:
        """
        Main training pipeline. The protocol includes the following steps:
        1) Initialization
        2) While loop of training
        3) Post processing after training
        
        Parameters:
        -----------
        clients: list
            List of MachineInfo object that contains the connect information of each client.
        
        Returns
        -------
        None
        """
        # Training initialization. Send initialization signal to all clients.
        if not hasattr(self, "_has_coordinator"):
            raise ValueError("The running client does not have coordinator addon!")
        phase = init_phase
        requests = self.coordinator.init_training_control()
        responses = self._exp_call_grpc_client(requests, is_parallel)
        requests, phase = self.coordinator.synchronous_control(responses, phase)

        # Training loop. parallel sending requests
        while self.coordinator.is_training_continue():
            responses = self._exp_call_grpc_client(requests, is_parallel)
            requests, phase = self.coordinator.synchronous_control(responses, phase)

        # Training process finish. Send finish signal to all clients.
        requests = self.coordinator.post_training_session()
        responses = self._exp_call_grpc_client(requests, is_parallel)

    def start_serve_termination_block(self):
        self.grpc_node.start_serve_termination_block(self.grpc_servicer)
