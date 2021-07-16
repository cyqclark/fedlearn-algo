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

# add core's path
import os
import sys
sys.path.append(os.getcwd())
#
from core.entity.common.message import RequestMessage, ResponseMessage
from core.entity.common.machineinfo import MachineInfo
from core.server.server import Server
from demos.custom_alg_demo.utils import Symbol
from demos.custom_alg_demo import utils
import copy
import socket
from typing import Dict

try:
    _LOCALHOST = socket.gethostbyname(socket.gethostname())
except socket.gaierror:
    # if failed to fetch 127.0.0.1, try 0.0.0.0
    _LOCALHOST = socket.gethostbyname("")


class KernelMethodServerError(ValueError):
    pass


class CustomDemoServer(Server):
    """customized client.

    This class is derived from the Client class, and simulated the client terminal.

    Attributes
    ----------
    machine_info : MachineInfo
        record current terminal ip, port, and token
    clients_info : List
        all clients' MachineInfo
    max_iter : int
        max iter number
    iter : int
        current iter number
    clients_token : List
        record clients' tokens
    dict_functions : dictionary
        record all training and inference functions

    Methods
    -------
    function_register(self)
        Define the correspondence between response message type and server processing function.
    init_training_control(request: RequestMessage)
        Send training initialization request to clients.
    req_tr_intermediate_comp_1(request: RequestMessage)
        intermediate computation stage 1.
    req_tr_intermediate_comp_2(request: RequestMessage)
        intermediate computation stage 2.
    req_tr_intermediate_comp_3(request: RequestMessage)
        intermediate computation stage 3.
    post_training_session(request: RequestMessage)
        Send finish signal to clients.   
    """

    def __init__(self, machine_info: MachineInfo):
        self.machine_info = machine_info
        self.clients_info = None
        self.max_iter = 3
        self.iter = 0
        self.clients_token = []
        self.dict_functions = {}
        self.function_registration()
        # variables for msg body validation
        self.msg_body_init_tr = utils.create_simulated_msg_body_strings_ints()
        self.msg_body_arrays = utils.create_simulated_msg_body_arrays()
        self.msg_body_en_bytes, self.private_key_out = utils.create_simulated_msg_body_encryption_bytes()
        self.msg_body_model_bytes = utils.create_simulated_msg_body_model_bytes()
        self.msg_body_ndarray_bytes = utils.create_simulated_msg_body_ndarray_bytes()

    def function_registration(self):
        """
        Define the correspondence between response message type and server processing function.
        """
        # funcs for training in server terminal
        self.dict_functions[Symbol.res_tr_init_finish] = self.req_tr_intermediate_comp_1
        self.dict_functions[Symbol.res_tr_intermediate_comp_finish_1] = self.req_tr_intermediate_comp_2
        self.dict_functions[Symbol.res_tr_intermediate_comp_finish_2] = self.req_tr_intermediate_comp_3
        self.dict_functions[Symbol.res_tr_intermediate_comp_finish_3] = self.req_tr_intermediate_comp_1
        self.dict_functions[Symbol.res_tr_finish] = self.post_training_session

    def get_next_phase(self, phase: str) -> str:
        """
        Transfer old phase of client to new phase of server
        """
        # only training
        if phase == "empty":
            next_phase = Symbol.res_tr_init_finish
        elif (phase == Symbol.res_tr_init_finish) or (phase == Symbol.res_tr_intermediate_comp_finish_3):
            next_phase = Symbol.res_tr_intermediate_comp_finish_1
        elif phase == Symbol.res_tr_intermediate_comp_finish_1:
            next_phase = Symbol.res_tr_intermediate_comp_finish_2
        elif phase == Symbol.res_tr_intermediate_comp_finish_2:
            next_phase = Symbol.res_tr_intermediate_comp_finish_3
        else:
            raise ValueError("Cannot find phase %s in both train and inference!" % phase)

        return next_phase

    # the following functions are used to control the customized alg workflow
    def init_training_control(self) -> Dict[MachineInfo, RequestMessage]:
        """ Init training control.

        Send training initialization request to clients.

        Parameters
        ----------

        Returns
        -------
        requests : dictionary
            all requests in server terminal.
        """
        print('custom server, init_training_control.')
        requests = {}
        for client_info in self.clients_info:
            print("client, ip: %s, port: %s, token: %s" % (client_info.ip, client_info.port, client_info.token))
            requests[client_info] = RequestMessage(sender=self.machine_info,
                                                   receiver=client_info,
                                                   body=copy.deepcopy(self.msg_body_init_tr),
                                                   phase_id=Symbol.req_tr_init)
        print("-------")
        print("")
        return requests

    # when get client's symbol: res_tr_loop_start_finish
    def req_tr_intermediate_comp_1(self, responses: Dict[MachineInfo, ResponseMessage]) -> \
            Dict[MachineInfo, RequestMessage]:
        """ stage 1.

        Send request to clients to compute intermediate results 1.

        Parameters
        ----------
        responses : dictionary
            all responses from client terminals.

        Returns
        -------
        requests : dictionary
            all requests in server terminal.
        """
        print('custom server, req_tr_intermediate_comp_1.')
        requests = {}
        for client_info, response in responses.items():
            utils.compare_two_msg_bodies_strings_ints(self.msg_body_init_tr, response.body)
            print("client, ip: %s, port: %s, token: %s" % (client_info.ip, client_info.port, client_info.token))
            request = RequestMessage(sender=self.machine_info,
                                     receiver=client_info,
                                     body=copy.deepcopy(self.msg_body_arrays),
                                     phase_id=Symbol.req_tr_intermediate_comp_1)
            requests[client_info] = request
        print("-------")
        print("")
        return requests

    # when get client's symbol: res_tr_intermediate_comp_finish_1
    def req_tr_intermediate_comp_2(self, responses: Dict[MachineInfo, ResponseMessage]) -> \
            Dict[MachineInfo, RequestMessage]:
        """ stage 2.

        Send request to clients to compute intermediate results 2.

        Parameters
        ----------
        responses : dictionary
            all responses from client terminals.

        Returns
        -------
        requests : dictionary
            all requests in server terminal.
        """
        print('custom server, req_tr_intermediate_comp_2.')
        requests = {}
        for client_info, response in responses.items():
            utils.compare_two_msg_bodies_arrays(self.msg_body_arrays, response.body)
            print("client, ip: %s, port: %s, token: %s" % (client_info.ip, client_info.port, client_info.token))
            request = RequestMessage(sender=self.machine_info,
                                     receiver=client_info,
                                     body=copy.deepcopy(self.msg_body_ndarray_bytes),
                                     phase_id=Symbol.req_tr_intermediate_comp_2)
            requests[client_info] = request
        print("-------")
        print("")
        return requests

    def req_tr_intermediate_comp_3(self, responses: Dict[MachineInfo, ResponseMessage]) -> \
            Dict[MachineInfo, RequestMessage]:
        """ stage 3.

        Send request to clients to compute intermediate results 3.

        Parameters
        ----------
        responses : dictionary
            all responses from client terminals.

        Returns
        -------
        requests : dictionary
            all requests in server terminal.
        """
        print('custom server, req_tr_intermediate_comp_3.')
        requests = {}
        for client_info, response in responses.items():
            utils.compare_two_msg_bodies_ndarray_bytes(self.msg_body_ndarray_bytes, response.body)
            print("client, ip: %s, port: %s, token: %s" % (client_info.ip, client_info.port, client_info.token))
            request = RequestMessage(sender=self.machine_info,
                                     receiver=client_info,
                                     body=copy.deepcopy(self.msg_body_init_tr),
                                     phase_id=Symbol.req_tr_intermediate_comp_3)
            requests[client_info] = request
        print("iter = %d" % self.iter)
        self.iter += 1
        print("-------")
        print("")
        return requests

    # the following is to override the derived functions.
    def is_training_continue(self) -> bool:
        if self.iter <= self.max_iter:
            return True
        else:
            return False

    def post_training_session(self) -> Dict[MachineInfo, RequestMessage]:
        """ finish stage.

        Send finish signal to clients.

        Parameters
        ----------

        Returns
        -------
        requests : dictionary
            all requests in server terminal.
        """
        requests = {}
        for client_info in self.clients_info:
            body = {'message': 'finish_training'}
            requests[client_info] = RequestMessage(sender=self.machine_info,
                                                   receiver=client_info,
                                                   body=body,
                                                   phase_id=Symbol.req_tr_finish)
        return requests

    def is_inference_continue(self) -> None:
        return

    def post_inference_session(self) -> None:
        return

    def init_inference_control(self) -> Dict[MachineInfo, RequestMessage]:
        """
        Send request to clients for inference initialization.
        """
        requests = {}
        for client_info in self.clients_info:
            requests[client_info] = RequestMessage(sender=self.machine_info,
                                                   receiver=client_info,
                                                   body=None,
                                                   phase_id=Symbol.req_infer_init)
        return requests


if __name__ == '__main__':
    # machine information
    master_info = MachineInfo(ip=_LOCALHOST, port='8890', token='master_machine')
    client_info1 = MachineInfo(ip=_LOCALHOST, port='8891', token='client_1')
    client_info2 = MachineInfo(ip=_LOCALHOST, port='8892', token='client_2')
    client_info3 = MachineInfo(ip=_LOCALHOST, port='8893', token='client_3')

    server = CustomDemoServer(master_info)
    server.clients_info = [client_info1, client_info2, client_info3]
    server.clients_token = [client_info1.token, client_info2.token, client_info3.token]

    server.training_pipeline("empty", server.clients_info)
