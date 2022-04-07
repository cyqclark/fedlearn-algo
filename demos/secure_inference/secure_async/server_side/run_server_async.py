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
import numpy as np
np.random.seed(2)

import os
import sys
sys.path.append(os.getcwd())
#
from core.entity.common.message import ResponseMessage
from core.entity.common.machineinfo import MachineInfo
from demos.secure_inference.secure_async.utils.grpc_async_communicator import AsyncGRPCCommunicator
import socket
from Secure_Server import Secure_Server
import json
import time

# global variables
_EPSILON = 1e-8
try:
    _LOCALHOST = socket.gethostbyname(socket.gethostname())
except socket.gaierror:
    # if failed to fetch 127.0.0.1, try 0.0.0.0
    _LOCALHOST = socket.gethostbyname("")

class Server(object):
    def __init__(self, server_info):
        #
        self.server_info = server_info
        self.machine_info = self.server_info
        self.sync_server = Secure_Server(server_info)

        #
        self.dict_functions = {}
        self.function_register()

    def function_register(self):
        self.dict_functions = self.sync_server.dict_functions
        self.dict_functions["init_comput_graph"] = self.init_comput_graph

    def process_queue(self, msg):
        phase_id = msg.phase_id
        if phase_id not in self.dict_functions.keys():
            raise ValueError("%s, Function %s is not implemented." % (self.__class__.__name__, phase_id))
        print(f'server run msg: {msg}')
        resp_msg_body = self.dict_functions[phase_id](msg)

        resp_msg = ResponseMessage(sender = msg.server_info,
                                   receiver = self.machine_info, #TODO: checkout why not msg.client_info
                                   body = resp_msg_body,  # numpy tensor
                                   phase_id=phase_id)
        return resp_msg


    def init_comput_graph(self, request):
        response_msg_body = {'graph': json.dumps(self.sync_server.compute_graph)}
        return response_msg_body


if __name__ == '__main__':
    # machine information
    server_info = MachineInfo(ip=_LOCALHOST, port='38892', token='simulated master for test')
    server = Server(server_info)
    server_async = AsyncGRPCCommunicator(server)
    server_async.start_grpc_message_processing()
    # stop
    # server_async.stop_grpc_node_receive_routine()

    s_seconds = 2000
    print("after a to b, sleep %s seconds." % s_seconds)
    time.sleep(s_seconds)