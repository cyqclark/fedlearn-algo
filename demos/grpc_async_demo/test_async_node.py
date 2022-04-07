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
from demos.grpc_async_demo.grpc_async_communicator import AsyncGRPCCommunicator
import copy
import numpy as np
import random
import socket
import time

# global variables
_EPSILON = 1e-8
try:
    _LOCALHOST = socket.gethostbyname(socket.gethostname())
except socket.gaierror:
    # if failed to fetch 127.0.0.1, try 0.0.0.0
    _LOCALHOST = socket.gethostbyname("")


def create_simulated_msg_body_strings_ints():
    # msg body init
    ts_msg_body_dict = dict()

    # strings and ints
    ts_msg_body_dict["multi_str"] = ["layer_1", "layer_2", "layer_3"]
    ts_msg_body_dict["multi_int"] = [random.randint(1, 9999), random.randint(1, 9999), random.randint(1, 9999)]
    ts_msg_body_dict["single_str"] = "layer_n"
    ts_msg_body_dict["single_int_0"] = np.random.randint(0, 9999, size=(1,), dtype=np.int32)[0]
    ts_msg_body_dict["single_int_1"] = np.random.randint(0, 9999, size=(1,), dtype=np.uint64)[0]

    return ts_msg_body_dict


def compare_two_msg_bodies_strings_ints(msg_body_1, msg_body_2, str_msg=""):
    # strings and ints
    msg_1_multi_str = msg_body_1["multi_str"]
    msg_2_multi_str = msg_body_2["multi_str"]
    for i in range(len(msg_1_multi_str)):
        assert (msg_1_multi_str[i] == msg_2_multi_str[i])

    msg_1_multi_ints = msg_body_1["multi_int"]
    msg_2_multi_ints = msg_body_2["multi_int"]
    for i in range(len(msg_1_multi_ints)):
        assert (msg_1_multi_ints[i] == msg_2_multi_ints[i])

    msg_1_single_str = msg_body_1["single_str"]
    msg_2_single_str = msg_body_2["single_str"]
    assert (msg_1_single_str == msg_2_single_str)

    msg_1_single_int_0 = msg_body_1["single_int_0"]
    msg_2_single_int_0 = msg_body_2["single_int_0"]
    assert (msg_1_single_int_0 == msg_2_single_int_0)

    msg_1_single_int_1 = msg_body_1["single_int_1"]
    msg_2_single_int_1 = msg_body_2["single_int_1"]
    assert (msg_1_single_int_1 == msg_2_single_int_1)

    # all items are matched.
    print("%s: strings_ints is OK." % str_msg)


class TerminalA(object):
    def __init__(self, master_info, client_info):
        #
        self.machine_info = master_info
        self.target_info = client_info
        #
        self.ts_msg_body = create_simulated_msg_body_strings_ints()
        #
        self.dict_functions = {}
        self.function_register()

    def function_register(self):
        self.dict_functions["res_a_1"] = self.response_for_res_a_1

    def process_queue(self, request):
        symbol = request.phase_id
        if symbol not in self.dict_functions.keys():
            raise ValueError("%s, Function %s is not implemented." % (self.__class__.__name__, symbol))
        response = self.dict_functions[symbol](request)
        return response

    def start_test(self):
        init_req = RequestMessage(sender=self.machine_info,
                                  receiver=self.target_info,
                                  body=copy.deepcopy(self.ts_msg_body),
                                  phase_id="req_a_1")
        return init_req
        
    def response_for_res_a_1(self, request):
        print('processing stage %s in terminal a.' % request.phase_id)

        compare_two_msg_bodies_strings_ints(self.ts_msg_body, request.body, "terminal a checks responsed msg body")

        sender_info = request.server_info
        receiver_info = request.client_info
        response = ResponseMessage(sender=sender_info,
                                   receiver=receiver_info,
                                   body={},
                                   phase_id="finish")
        return response


class TerminalB(object):
    def __init__(self, master_info, client_info):
        #
        self.machine_info = master_info
        self.target_info = client_info
        #
        self.dict_functions = {}
        self.function_register()

    def function_register(self):
        self.dict_functions["req_a_1"] = self.response_for_req_a_1

    def process_queue(self, request):
        symbol = request.phase_id
        print(symbol)
        print(self.dict_functions.keys())
        if symbol not in self.dict_functions.keys():
            raise ValueError("%s, Function %s is not implemented." % (self.__class__.__name__, symbol))
        response = self.dict_functions[symbol](request)
        return response

    # the following functsions are used to control the customized alg workflow
    def response_for_req_a_1(self, request):
        print('processing stage %s in terminal b.' % request.phase_id)
        sender_info = request.server_info
        receiver_info = request.client_info
        response = ResponseMessage(sender=sender_info,
                                   receiver=receiver_info,
                                   body=copy.deepcopy(request.body),
                                   phase_id="res_a_1")
        return response


if __name__ == '__main__':
    # machine information
    t_a_info = MachineInfo(ip=_LOCALHOST, port='38890', token='simulated master for test')
    t_b_info = MachineInfo(ip=_LOCALHOST, port='38891', token='simulated client for test')

    # a
    terminal_a = TerminalA(t_a_info, t_b_info)
    async_grpc_comm_a = AsyncGRPCCommunicator(terminal_a, t_a_info)
    async_grpc_comm_a.start_grpc_message_processing()
    
    # b
    terminal_b = TerminalB(t_b_info, t_a_info)
    async_grpc_comm_b = AsyncGRPCCommunicator(terminal_b, t_b_info)
    async_grpc_comm_b.start_grpc_message_processing()

    # a --> b
    print("0, a --> b")
    init_req = terminal_a.start_test()
    print("1, a --> b")
    async_grpc_comm_a.send_message(init_req)
    print("2, a --> b")

    s_seconds = 1
    print("after a to b, sleep %s seconds." % s_seconds)
    time.sleep(1)

    # stop
    async_grpc_comm_b.stop_grpc_node_receive_routine()
    async_grpc_comm_a.stop_grpc_node_receive_routine()
