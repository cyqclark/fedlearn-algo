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
"""
Coordinator base class
"""

from core.entity.common.machineinfo import MachineInfo
from core.entity.common.message import RequestMessage, ResponseMessage
from core.grpc_comm.grpc_client import send_request
from core.server.server import SendingThread, send_grpc_request, call_grpc_client

from abc import ABC, abstractmethod
from threading import Thread
from typing import Dict, List


class Coordinator(ABC):
    """
    Base class of server.
    The abstract functions need to be instantiated in each algorithm.
    The defined control and pipeline function are shared by vertical federated learning algorithm examples.
    """
    #def __init__(self):
    #    self.__has_coordinator = True
    #    return None

    @property
    def control_functions(self):
        """
        Dictionary of functions that store the function mapping in both training and
        inference as <phase_id: function>.
        """
        return self._control_functions

    @control_functions.setter
    def control_functions(self, value):
        if not isinstance(value, dict):
            raise ValueError("Funcion mapping must be a dictionary!")
        self._control_functions = value

    @abstractmethod
    def init_training_control(self) -> Dict[MachineInfo, RequestMessage]:
        """
        Training initialization.

        Returns
        -------
        requests: Dict
           The dictionary containing request message to clients.
        """

    @abstractmethod
    def get_next_phase(self, phase: str) -> str:
        """
        Get next phase given the current phase,
        this function works in both training part and inference part
        """

    @abstractmethod
    def is_training_continue(self) -> bool:
        """
        Check if training continues or not.

        Returns
        -------
        flag: bool
            The flag which shows if training continues. True = continue; False = end.
        """

    @abstractmethod
    def is_inference_continue(self) -> bool:
        """
        Check if inference continues or not.

        Returns
        -------
        flag: bool
            The flag which shows if inference continues. True = continue; False = end.
        """

    @abstractmethod
    def post_training_session(self) -> Dict[MachineInfo, RequestMessage]:
        """
        Post processing on sever size after training is finished.
        
        Returns
        -------
        None
        """

    @abstractmethod
    def init_inference_control(self) -> Dict[MachineInfo, RequestMessage]:
        """
        Inference initialization.
        :return: Dictionary containing message to clients that request initialization.
        """

    @abstractmethod
    def post_inference_session(self) -> None:
        """
        Finish the inference process.
        :return: None.
        """
        
    def inference_pipeline(self, init_phase:str, clients: List[MachineInfo], is_parallel=True) -> None:
        """
        This function defines the inference pipeline.
        The inference pipeline has two steps - initialization and local computation.
        The function is shared by all vertical federated learning examples.

        :param: clients: list containing client machines' information.
        :return: None.
        """
        phase = init_phase
        requests = self.init_inference_control()
        responses = {}
        while self.is_inference_continue():
            responses = call_grpc_client(requests, is_parallel)
            requests, phase = self.synchronous_control(responses, phase)
        self.post_inference_session()
        return

    def synchronous_control(self, responses: Dict[MachineInfo, ResponseMessage], phase: str) -> RequestMessage:
        """
        Synchronous control function: this function extract the target training phase function
            in the function mapping and run the target training function to generate
            requests. Synchronous control requires all the phase id of client responses are
            the same.

        Parameters
        ----------
        responses: dict
            The dictionary which contains the response messages from the clients.
            The dictionary is stored as <MachineInfo: ResponseMessage>.
        
        Returns
        -------
        requests: dict
            The dictionary of requests to clients.
            The dictionary is stored as <MachineInfo: RequestMessage>.
        """
        symbol = None
        for client_info, response in responses.items():
            if symbol is None:
                symbol = response.phase_id
            else:
                if not symbol == responses[client_info].phase_id:
                    raise ValueError("Phase id of client responses are not aligned")
        
        new_phase = self.get_next_phase(phase)
        if new_phase not in self.control_functions.keys():
            raise ValueError("Cannot find phase %s in function dictionary"%new_phase)
        requests = self.control_functions[new_phase](responses)
        return requests, new_phase

