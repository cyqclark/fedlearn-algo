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
from core.client.client import Client
from core.entity.common.message import RequestMessage, ResponseMessage
from core.entity.common.machineinfo import MachineInfo
from demos.custom_alg_demo.utils import Symbol
import argparse
import socket
from typing import List

try:
    _LOCALHOST = socket.gethostbyname(socket.gethostname())
except socket.gaierror:
    # if failed to fetch 127.0.0.1, try 0.0.0.0
    _LOCALHOST = socket.gethostbyname("")


class ClientError(ValueError):
    pass


class CustomDemoClient(Client):
    """customized client.

    This class is derived from the Client class, and simulated the client terminal.

    Attributes
    ----------
    machine_info : MachineInfo
        record current terminal ip, port, and token
    dict_functions : dictionary
        record all training and inference functions

    Methods
    -------
    function_register(self)
        register all functions in the training loop.
    train_init(request: RequestMessage)
        Prepare training.
    intermediate_comp_1(request: RequestMessage)
        intermediate computation stage 1.
    intermediate_comp_2(request: RequestMessage)
        intermediate computation stage 2.
    intermediate_comp_3(request: RequestMessage)
        intermediate computation stage 3.
    training_finish(request: RequestMessage)
        Finish training.    
    """

    def __init__(self, machine_info: MachineInfo):
        # init grpc_node and grpc_servicer
        super(CustomDemoClient, self).__init__()

        self.machine_info = machine_info
        self.is_active = False
        self.dict_functions = {}
        self.function_register()

        self.grpc_node.set_machine_info(self.machine_info)
        self.grpc_servicer.set_msg_processor(self)

    def function_register(self):
        """ training and inference functions registration.

        register all functions in the training loop.

        Parameters
        ----------

        Returns
        -------
        """
        # funcs for training in client terminal
        self.dict_functions[Symbol.req_tr_init] = self.train_init
        self.dict_functions[Symbol.req_tr_intermediate_comp_1] = self.intermediate_comp_1
        self.dict_functions[Symbol.req_tr_intermediate_comp_2] = self.intermediate_comp_2
        self.dict_functions[Symbol.req_tr_intermediate_comp_3] = self.intermediate_comp_3
        self.dict_functions[Symbol.req_tr_finish] = self.training_finish

    def train_init(self, request: RequestMessage) -> ResponseMessage:
        """ Init stage.

        Prepare training.

        Parameters
        ----------
        request : RequestMessage
            a request message to simulate the input request.

        Returns
        -------
        response : ResponseMessage
            a response message to simulate the output response.
        """
        print('custom client, train_init, ip: %s, port: %s, token: %s' % (self.machine_info.ip,
                                                                          self.machine_info.port,
                                                                          self.machine_info.token))
        sender_info = request.server_info
        receiver_info = request.client_info
        response = ResponseMessage(sender=sender_info,
                                   receiver=receiver_info,
                                   body=request.body,
                                   phase_id=Symbol.res_tr_init_finish)
        print("***********")
        print("")
        return response

    def intermediate_comp_1(self, request: RequestMessage) -> ResponseMessage:
        """ stage 1.

        intermediate computation stage 1.

        Parameters
        ----------
        request : RequestMessage
            a request message to simulate the input request.

        Returns
        -------
        response : ResponseMessage
            a response message to simulate the output response.
        """
        print('custom client, intermediate_comp_1, ip: %s, port: %s, token: %s' % (self.machine_info.ip,
                                                                                   self.machine_info.port,
                                                                                   self.machine_info.token))
        sender_info = request.server_info
        receiver_info = request.client_info
        response = ResponseMessage(sender=sender_info,
                                   receiver=receiver_info,
                                   body=request.body,
                                   phase_id=Symbol.res_tr_intermediate_comp_finish_1)
        print("***********")
        print("")
        return response

    def intermediate_comp_2(self, request: RequestMessage) -> ResponseMessage:
        """ stage 2.

        intermediate computation stage 2.

        Parameters
        ----------
        request : RequestMessage
            a request message to simulate the input request.

        Returns
        -------
        response : ResponseMessage
            a response message to simulate the output response.
        """
        print('custom client, intermediate_comp_2, ip: %s, port: %s, token: %s' % (self.machine_info.ip,
                                                                                   self.machine_info.port,
                                                                                   self.machine_info.token))
        sender_info = request.server_info
        receiver_info = request.client_info
        response = ResponseMessage(sender=sender_info,
                                   receiver=receiver_info,
                                   body=request.body,
                                   phase_id=Symbol.res_tr_intermediate_comp_finish_2)
        print("***********")
        print("")
        return response

    def intermediate_comp_3(self, request: RequestMessage) -> ResponseMessage:
        """ stage 3.

        intermediate computation stage 3.

        Parameters
        ----------
        request : RequestMessage
            a request message to simulate the input request.

        Returns
        -------
        response : ResponseMessage
            a response message to simulate the output response.
        """
        print('custom client, intermediate_comp_3, ip: %s, port: %s, token: %s' % (self.machine_info.ip,
                                                                                   self.machine_info.port,
                                                                                   self.machine_info.token))
        sender_info = request.server_info
        receiver_info = request.client_info
        response = ResponseMessage(sender=sender_info,
                                   receiver=receiver_info,
                                   body=request.body,
                                   phase_id=Symbol.res_tr_intermediate_comp_finish_3)
        print("***********")
        print("")
        return response

    def training_finish(self, request: RequestMessage) -> ResponseMessage:
        """ finish stage.

        Finish training.

        Parameters
        ----------
        request : RequestMessage
            a request message to simulate the input request.

        Returns
        -------
        response : ResponseMessage
            a response message to simulate the output response.
        """
        print('custom client, training_finish, ip: %s, port: %s, token: %s' % (self.machine_info.ip,
                                                                               self.machine_info.port,
                                                                               self.machine_info.token))
        sender_info = request.server_info
        receiver_info = request.client_info
        print("save model!!!")
        response = ResponseMessage(sender=sender_info,
                                   receiver=receiver_info,
                                   body=None,
                                   phase_id=Symbol.res_tr_finish)
        print("***********")
        print("")
        return response

    def inference_init(self, request: RequestMessage) -> ResponseMessage:
        return request.copy()

    def load_data(self, data_path: str, feature_names: List, label_name: List = None):
        """ loading data.

        Just demo, and set the active label as "True".

        Parameters
        ----------
        data_path : str
            data's path.
        feature_names : List
            feature names used in training.
        label_name : List
            label names used in training.

        Returns
        -------
        None
        """
        print("No data, just demo.")
        if label_name is not None:
            self.is_active = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # machine info
    parser.add_argument('-I', '--ip', type=str, help="string of server ip, exp:%s" % _LOCALHOST)
    parser.add_argument('-P', '--port', type=str, help="string of server port, exp: 8890")
    parser.add_argument('-T', '--token', type=str, help="string of server name, exp: client_1")

    args = parser.parse_args()
    # TODO
    print("ip: %s" % args.ip)
    print("port: %s" % args.port)
    print("token: %s" % args.token)

    # create client object
    client_info = MachineInfo(ip=args.ip, port=args.port, token=args.token)
    client = CustomDemoClient(client_info)

    # training
    client.load_data(data_path="no path, just testing", feature_names=["no feature name, just testing"])
    # # create gRPC service
    # serve(client)

    client.start_serve_termination_block()
