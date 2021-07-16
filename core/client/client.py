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
from core.grpc_comm.grpc_converter import grpc_msg_to_common_msg, common_msg_to_grpc_msg
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
