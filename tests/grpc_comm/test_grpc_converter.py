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
from core.entity.common.message import RequestMessage
from core.entity.common.machineinfo import MachineInfo
from core.grpc_comm.grpc_converter import common_dict_msg_to_arrays, arrays_to_common_dict_msg, \
    create_grpc_message, parse_grpc_message, common_msg_to_grpc_msg, grpc_msg_to_common_msg
import copy
from io import BytesIO
import logging
import numpy as np
import pickle
import random
import socket
from typing import cast
import unittest

# global variables
_EPSILON = 1e-8
try:
    _LOCALHOST = socket.gethostbyname(socket.gethostname())
except socket.gaierror:
    # if failed to fetch 127.0.0.1, try 0.0.0.0
    _LOCALHOST = socket.gethostbyname("")


def numpy_array_to_bytes(numpy_array):
    bytes_io = BytesIO()
    np.save(bytes_io, numpy_array, allow_pickle=False)
    return bytes_io.getvalue()


def bytes_to_numpy_array(tensor):
    bytes_io = BytesIO(tensor)
    numpy_array_deserialized = np.load(bytes_io, allow_pickle=False)
    return cast(np.ndarray, numpy_array_deserialized)


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


def create_simulated_msg_body_arrays():
    # msg body init
    ts_msg_body_dict = dict()

    # arrays
    val = np.random.randn()
    vec = np.random.randn(6)
    mx = np.random.randn(4, 4)
    ts_msg_body_dict["val_0"] = [val, val*2, val*3]
    ts_msg_body_dict["val_1"] = val*2
    ts_msg_body_dict["vec_0"] = vec
    ts_msg_body_dict["vec_1"] = [vec, vec*2]
    ts_msg_body_dict["mx_0"] = [mx, mx*2]
    ts_msg_body_dict["mx_1"] = mx

    return ts_msg_body_dict


def create_simulated_msg_body_model_bytes():
    # msg body init
    ts_msg_body_dict = dict()

    # models
    w_avg = np.random.randn(16, 16)
    b_avg = np.random.randn(16, 1)
    parameters = {'w': w_avg, 'b': b_avg}
    bytes_buff_1 = pickle.dumps(parameters)

    w_avg = np.random.randn(16, 16)
    b_avg = np.random.randn(16, 1)
    parameters = {'w': w_avg, 'b': b_avg}
    bytes_buff_2 = pickle.dumps(parameters)

    ts_msg_body_dict["model_bytes_0"] = [bytes_buff_1, bytes_buff_2]
    ts_msg_body_dict["model_bytes_1"] = bytes_buff_1

    return ts_msg_body_dict


def create_simulated_msg_body_ndarray_bytes():
    # msg body init
    ts_msg_body_dict = dict()

    # numpy_array to bytes
    ts_msg_body_dict["numpy_array_bytes_0"] = [numpy_array_to_bytes(np.random.randn(64, 64).astype(np.float32)),
                                               numpy_array_to_bytes(np.random.randn(64, 64).astype(np.float32)),
                                               numpy_array_to_bytes(np.random.randn(64, 64).astype(np.float32))]
    ts_msg_body_dict["numpy_array_bytes_1"] = numpy_array_to_bytes(np.random.randn(256, 256).astype(np.float32))

    return ts_msg_body_dict


class TestGRPCConverter(unittest.TestCase):
    def test_common_dict_msg_to_arrays_arrays_to_common_dict_msg(self):
        # msg body
        ts_msg_body = create_simulated_msg_body_arrays()

        # msg body to arrays, arrays to msg body
        dict_buffs, dict_notes = common_dict_msg_to_arrays(ts_msg_body)
        returned_msg_body = arrays_to_common_dict_msg(dict_buffs, dict_notes)

        # compare msg body
        msg_1_val_0 = ts_msg_body["val_0"]
        msg_2_val_0 = returned_msg_body["val_0"]
        for i in range(len(msg_1_val_0)):
            temp1 = msg_1_val_0[i]
            temp2 = msg_2_val_0[i]
            self.assertLess(np.linalg.norm(temp1 - temp2), _EPSILON)

        msg_1_val_1 = ts_msg_body["val_1"]
        msg_2_val_1 = returned_msg_body["val_1"]
        self.assertLess(np.linalg.norm(msg_1_val_1 - msg_2_val_1), _EPSILON)

        msg_1_vec_0 = ts_msg_body["vec_0"]
        msg_2_vec_0 = returned_msg_body["vec_0"]
        self.assertLess(np.linalg.norm(msg_1_vec_0 - msg_2_vec_0), _EPSILON)

        msg_1_vec_1 = ts_msg_body["vec_1"]
        msg_2_vec_1 = returned_msg_body["vec_1"]
        for i in range(len(msg_1_vec_1)):
            temp1 = msg_1_vec_1[i]
            temp2 = msg_2_vec_1[i]
            self.assertLess(np.linalg.norm(temp1 - temp2), _EPSILON)

        msg_1_mx_0 = ts_msg_body["mx_0"]
        msg_2_mx_0 = returned_msg_body["mx_0"]
        for i in range(len(msg_1_mx_0)):
            temp1 = msg_1_mx_0[i]
            temp2 = msg_2_mx_0[i]
            self.assertLess(np.linalg.norm(temp1 - temp2), _EPSILON)

        msg_1_mx_1 = ts_msg_body["mx_1"]
        msg_2_mx_1 = returned_msg_body["mx_1"]
        self.assertLess(np.linalg.norm(msg_1_mx_1 - msg_2_mx_1), _EPSILON)

    def test_create_grpc_message_parse_grpc_message(self):
        # msg body
        ts_msg_body = create_simulated_msg_body_model_bytes()

        # machine information
        master_info = MachineInfo(ip=_LOCALHOST, port='8890', token='master_machine')
        client_info = MachineInfo(ip=_LOCALHOST, port='8891', token='client_machine')

        # phase id
        phase_id = str(random.randint(1, 9999))

        # create buffs and notes from a simulated msg body
        dict_buffs, dict_notes = common_dict_msg_to_arrays(ts_msg_body)
        # get a grpc_message
        grpc_message = create_grpc_message(master_info, 
                                           client_info, 
                                           dict_buffs, 
                                           dict_notes, 
                                           phase_id)
        # parse a grpc_message
        re_sender, re_receiver, re_phase_num, re_dict_buffs, re_dict_notes = parse_grpc_message(grpc_message)
        # restore a msg body for comparison
        returned_msg_body = arrays_to_common_dict_msg(re_dict_buffs, re_dict_notes)

        # compare machine info and phase id
        self.assertEqual(str(master_info), str(re_sender))
        self.assertEqual(str(client_info), str(re_receiver))
        self.assertEqual(phase_id, re_phase_num)

        # compare msg body
        msg_1_model_bytes_0 = ts_msg_body["model_bytes_0"]
        msg_2_model_bytes_0 = returned_msg_body["model_bytes_0"]
        for i in range(len(msg_1_model_bytes_0)):
            temp1 = pickle.loads(msg_1_model_bytes_0[i])
            temp2 = pickle.loads(msg_2_model_bytes_0[i])
            msg_1_w = temp1["w"]
            msg_1_b = temp1["b"]
            msg_2_w = temp2["w"]
            msg_2_b = temp2["b"]
            self.assertLess(np.linalg.norm(msg_1_w - msg_2_w), _EPSILON)
            self.assertLess(np.linalg.norm(msg_1_b - msg_2_b), _EPSILON)

        temp1 = pickle.loads(ts_msg_body["model_bytes_1"])
        temp2 = pickle.loads(returned_msg_body["model_bytes_1"])
        msg_1_w = temp1["w"]
        msg_1_b = temp1["b"]
        msg_2_w = temp2["w"]
        msg_2_b = temp2["b"]
        self.assertLess(np.linalg.norm(msg_1_w - msg_2_w), _EPSILON)
        self.assertLess(np.linalg.norm(msg_1_b - msg_2_b), _EPSILON)

    def test_common_msg_to_grpc_msg_grpc_msg_to_common_msg(self):
        # msg body
        ts_msg_body = create_simulated_msg_body_ndarray_bytes()

        # machine information
        master_info = MachineInfo(ip=_LOCALHOST, port='8890', token='master_machine')
        client_info = MachineInfo(ip=_LOCALHOST, port='8891', token='client_machine')

        # phase id
        phase_id = str(random.randint(1, 9999))

        # simulated request from master
        request = RequestMessage(sender=master_info, 
                                 receiver=client_info,
                                 body=copy.deepcopy(ts_msg_body),
                                 phase_id=phase_id)

        # req common msg to grpc msg, and grpc msg to req common msg
        returned_request = grpc_msg_to_common_msg(common_msg_to_grpc_msg(request), comm_req_res=0)

        # compare machine info and phase id
        self.assertEqual(str(master_info), str(returned_request.server_info))
        self.assertEqual(str(client_info), str(returned_request.client_info))
        self.assertEqual(phase_id, returned_request.phase_id)

        # compare msg body
        msg_1_numpy_array_bytes_0 = ts_msg_body["numpy_array_bytes_0"]
        msg_2_numpy_array_bytes_0 = returned_request.body["numpy_array_bytes_0"]
        for i in range(len(msg_1_numpy_array_bytes_0)):
            temp1 = bytes_to_numpy_array(msg_1_numpy_array_bytes_0[i])
            temp2 = bytes_to_numpy_array(msg_2_numpy_array_bytes_0[i])
            self.assertLess(np.linalg.norm(temp1 - temp2), _EPSILON)

        temp1 = bytes_to_numpy_array(ts_msg_body["numpy_array_bytes_1"])
        temp2 = bytes_to_numpy_array(returned_request.body["numpy_array_bytes_1"])
        self.assertLess(np.linalg.norm(temp1 - temp2), _EPSILON)


if __name__ == '__main__':
    logging.info("Test grpc_converter ...")
    unittest.main()
