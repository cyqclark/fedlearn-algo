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
from core.grpc_comm.grpc_client import send_request
from core.grpc_comm.grpc_converter import grpc_msg_to_common_msg, common_msg_to_grpc_msg
from core.proto.transmission_pb2 import ReqResMessage
from core.proto.transmission_pb2_grpc import TransmissionServicer, add_TransmissionServicer_to_server
from concurrent import futures
import copy
import grpc
from io import BytesIO
import logging
import numpy as np
from phe import paillier
import pickle
import random
import socket
from typing import cast
import unittest   # The test framework


# global variables
_EPSILON = 1e-8
_MAX_MESSAGE_LENGTH = 1 << 30
try:
    _LOCALHOST = socket.gethostbyname(socket.gethostname())
except socket.gaierror:
    # if failed to fetch 127.0.0.1, try 0.0.0.0
    _LOCALHOST = socket.gethostbyname("")


# Encrypt vector
def encryptVectorPaillier(vector, public_key):
    return [public_key.encrypt(x) for x in vector]


# Encrypt matrix
def encryptMatrixPaillier(matrix, public_key):
    return [encryptVectorPaillier(vector, public_key) for vector in matrix]


# Decrypt vector
def decryptVectorPaillier(vector, private_key):
    return np.array([private_key.decrypt(x) for x in vector])


# Decrypt matrix
def decryptMatrixPaillier(matrix, private_key):
    return np.array([decryptVectorPaillier(vector, private_key) for vector in matrix])


def compare_machine_info(machine_info_1, machine_info_2):
    assert (machine_info_1.ip == machine_info_2.ip)
    assert (machine_info_1.port == machine_info_2.port)
    assert (machine_info_1.token == machine_info_2.token)


def numpy_array_to_bytes(numpy_array):
    bytes_io = BytesIO()
    np.save(bytes_io, numpy_array, allow_pickle=False)
    return bytes_io.getvalue()


def bytes_to_numpy_array(tensor):
    bytes_io = BytesIO(tensor)
    numpy_array_deserialized = np.load(bytes_io, allow_pickle=False)
    return cast(np.ndarray, numpy_array_deserialized)


# the msg body is a dictionary.
# dict_keys(['multi_str', 'multi_int', 'single_str', 'single_int_0', 'single_int_1',
#            'val_0', 'val_1', 'vec_0', 'vec_1', 'mx_0', 'mx_1',
#            'encryption_bytes_0', 'encryption_bytes_1', 
#            'model_bytes_0', 'model_bytes_1', 
#            'numpy_array_bytes_0', 'numpy_array_bytes_1'])
def create_ts_msg_body_for_arrays_encryption_models():
    """ create a msg body.

    create an example of msg body to validate gRPC communication.

    Parameters
    ----------

    Returns
    -------
    ts_msg_body_dict : dictionary
        simulated msg body.
    private_key_out : PaillierPrivateKey
        a Paillier private key for further usage
    """
    # msg body init
    ts_msg_body_dict = dict()

    # strings and ints
    ts_msg_body_dict["multi_str"] = ["layer_1", "layer_2", "layer_3"]
    ts_msg_body_dict["multi_int"] = [random.randint(1, 9999), random.randint(1, 9999), random.randint(1, 9999)]
    ts_msg_body_dict["single_str"] = "layer_n"
    ts_msg_body_dict["single_int_0"] = np.random.randint(0, 9999, size=(1,), dtype=np.int32)[0]
    ts_msg_body_dict["single_int_1"] = np.random.randint(0, 9999, size=(1,), dtype=np.uint64)[0]

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

    # encryption
    val = np.random.randn()
    vec = np.random.randn(100)
    mx = np.random.randn(10, 10)
    public_key, private_key_out = paillier.generate_paillier_keypair()

    paillier_val = public_key.encrypt(val)
    paillier_vec = encryptVectorPaillier(vec, public_key)
    paillier_mx = encryptMatrixPaillier(mx, public_key)
    encrypt_dict = {"encrypt_val_0": paillier_val,
                    "encrypt_vec_0": paillier_vec,
                    "encrypt_mx_0": paillier_mx}
    bytes_encrypt_1 = pickle.dumps(encrypt_dict)

    paillier_val = public_key.encrypt(val*2)
    paillier_vec = encryptVectorPaillier(vec*2, public_key)
    paillier_mx = encryptMatrixPaillier(mx*2, public_key)
    encrypt_dict = {"encrypt_val_0": paillier_val,
                    "encrypt_vec_0": paillier_vec,
                    "encrypt_mx_0": paillier_mx}
    bytes_encrypt_2 = pickle.dumps(encrypt_dict)

    ts_msg_body_dict["encryption_bytes_0"] = bytes_encrypt_1
    ts_msg_body_dict["encryption_bytes_1"] = [bytes_encrypt_1, bytes_encrypt_2]

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

    # numpy_array to bytes
    ts_msg_body_dict["numpy_array_bytes_0"] = [numpy_array_to_bytes(np.random.randn(64, 64).astype(np.float32)),
                                               numpy_array_to_bytes(np.random.randn(64, 64).astype(np.float32)),
                                               numpy_array_to_bytes(np.random.randn(64, 64).astype(np.float32))]
    ts_msg_body_dict["numpy_array_bytes_1"] = numpy_array_to_bytes(np.random.randn(256, 256).astype(np.float32))

    return ts_msg_body_dict, private_key_out


class SimulatedGRPCServicer(TransmissionServicer):
    def __init__(self, machine_info: MachineInfo):
        super().__init__()
        self.machine_info = machine_info

    def comm(self, grpc_req_msg: ReqResMessage, context) -> ReqResMessage:
        common_req_msg = grpc_msg_to_common_msg(grpc_req_msg)

        # message processing simulation
        # this demo has no process_request
        common_res_msg = ResponseMessage(sender=common_req_msg.client_info, 
                                         receiver=common_req_msg.server_info,
                                         body=common_req_msg.body,
                                         phase_id=common_req_msg.phase_id+"_"+self.machine_info.token)

        return common_msg_to_grpc_msg(common_res_msg)


class SimulatedGRPCServe(object):
    def __init__(self, grpc_servicer):
        options = [
            ('grpc.max_send_message_length', _MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', _MAX_MESSAGE_LENGTH),
        ]
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
        add_TransmissionServicer_to_server(grpc_servicer, self.server)
        self.server.add_insecure_port("%s:%s" % (grpc_servicer.machine_info.ip, grpc_servicer.machine_info.port))
        # TODO: using debug level (DebugOutput), not print, to display information
        print("---------------")
        print("starting %s:%s" % (grpc_servicer.machine_info.ip, grpc_servicer.machine_info.port))
        #
        self.server.start()
        self.is_serve_running = True

    def stop_receive_message(self):
        self.server.stop(None)
        self.is_serve_running = False


class TestGRPCClient(unittest.TestCase):
    def test_send_request(self):
        # create a simulated msg body (it has all the supported styles of item in the msg body dict)
        ts_msg_body, private_key = create_ts_msg_body_for_arrays_encryption_models()

        # machine information
        master_info = MachineInfo(ip=_LOCALHOST, port='8890', token='')
        client_info = MachineInfo(ip=_LOCALHOST, port='38891', token='simulated client for test')

        # start serve
        grcp_servicer = SimulatedGRPCServicer(client_info)
        grpc_serve = SimulatedGRPCServe(grcp_servicer)
        print("grpc serve is running: %s" % str(grpc_serve.is_serve_running))

        # request
        request = RequestMessage(sender=master_info, 
                                 receiver=client_info,
                                 body=copy.deepcopy(ts_msg_body),
                                 phase_id=str(random.randint(1, 9999)))

        # send request and receive 
        print("sending to %s, and phase is: %s." % (client_info.token, request.phase_id))
        response = send_request(request)
        print("receiving response...")

        # stop
        grpc_serve.stop_receive_message()
        print("grpc serve is running: %s" % str(grpc_serve.is_serve_running))

        # compare machine information
        self.assertEqual(master_info.ip, response.server_info.ip)
        self.assertEqual(master_info.port, response.server_info.port)
        self.assertEqual(master_info.token, response.server_info.token)
        self.assertEqual(client_info.ip, response.client_info.ip)
        self.assertEqual(client_info.port, response.client_info.port)
        self.assertEqual(client_info.token, response.client_info.token)
        print("machine info match.")
        # display req and res phase
        print("req phase: %s, req body id: %d" % (request.phase_id, id(request.body)))
        print("res phase: %s, res body id: %d" % (response.phase_id, id(response.body)))

        # compare msg body
        print("res phase num: %s" % response.phase_id)
        print("org req body id: %d, res body id: %d" % (id(ts_msg_body), id(response.body)))
        msg_body_1 = ts_msg_body
        private_key_1 = private_key
        msg_body_2 = response.body
        private_key_2 = private_key

        # strings and ints
        msg_1_multi_str = msg_body_1["multi_str"]
        msg_2_multi_str = msg_body_2["multi_str"]
        for i in range(len(msg_1_multi_str)):
            self.assertEqual(msg_1_multi_str[i], msg_2_multi_str[i])

        msg_1_multi_ints = msg_body_1["multi_int"]
        msg_2_multi_ints = msg_body_2["multi_int"]
        for i in range(len(msg_1_multi_ints)):
            self.assertEqual(msg_1_multi_ints[i], msg_2_multi_ints[i])

        msg_1_single_str = msg_body_1["single_str"]
        msg_2_single_str = msg_body_2["single_str"]
        self.assertEqual(msg_1_single_str, msg_2_single_str)

        msg_1_single_int_0 = msg_body_1["single_int_0"]
        msg_2_single_int_0 = msg_body_2["single_int_0"]
        self.assertEqual(msg_1_single_int_0, msg_2_single_int_0)

        msg_1_single_int_1 = msg_body_1["single_int_1"]
        msg_2_single_int_1 = msg_body_2["single_int_1"]
        self.assertEqual(msg_1_single_int_1, msg_2_single_int_1)

        # arrays
        msg_1_val_0 = msg_body_1["val_0"]
        msg_2_val_0 = msg_body_2["val_0"]
        for i in range(len(msg_1_val_0)):
            temp1 = msg_1_val_0[i]
            temp2 = msg_2_val_0[i]
            self.assertLess(np.linalg.norm(temp1 - temp2), _EPSILON)

        msg_1_val_1 = msg_body_1["val_1"]
        msg_2_val_1 = msg_body_2["val_1"]
        self.assertLess(np.linalg.norm(msg_1_val_1 - msg_2_val_1), _EPSILON)

        msg_1_vec_0 = msg_body_1["vec_0"]
        msg_2_vec_0 = msg_body_2["vec_0"]
        self.assertLess(np.linalg.norm(msg_1_vec_0 - msg_2_vec_0), _EPSILON)

        msg_1_vec_1 = msg_body_1["vec_1"]
        msg_2_vec_1 = msg_body_2["vec_1"]
        for i in range(len(msg_1_vec_1)):
            temp1 = msg_1_vec_1[i]
            temp2 = msg_2_vec_1[i]
            self.assertLess(np.linalg.norm(temp1 - temp2), _EPSILON)

        msg_1_mx_0 = msg_body_1["mx_0"]
        msg_2_mx_0 = msg_body_2["mx_0"]
        for i in range(len(msg_1_mx_0)):
            temp1 = msg_1_mx_0[i]
            temp2 = msg_2_mx_0[i]
            self.assertLess(np.linalg.norm(temp1 - temp2), _EPSILON)

        msg_1_mx_1 = msg_body_1["mx_1"]
        msg_2_mx_1 = msg_body_2["mx_1"]
        self.assertLess(np.linalg.norm(msg_1_mx_1 - msg_2_mx_1), _EPSILON)

        # encryption
        msg_1_bytes_en_0 = pickle.loads(msg_body_1["encryption_bytes_0"])
        msg_1_bytes_en_0_mx = decryptMatrixPaillier(msg_1_bytes_en_0["encrypt_mx_0"], private_key_1)
        msg_1_bytes_en_0_vec = decryptVectorPaillier(msg_1_bytes_en_0["encrypt_vec_0"], private_key_1)
        msg_1_bytes_en_0_val = private_key_1.decrypt(msg_1_bytes_en_0["encrypt_val_0"])
        msg_2_bytes_en_0 = pickle.loads(msg_body_2["encryption_bytes_0"])
        msg_2_bytes_en_0_mx = decryptMatrixPaillier(msg_2_bytes_en_0["encrypt_mx_0"], private_key_2)
        msg_2_bytes_en_0_vec = decryptVectorPaillier(msg_2_bytes_en_0["encrypt_vec_0"], private_key_2)
        msg_2_bytes_en_0_val = private_key_2.decrypt(msg_2_bytes_en_0["encrypt_val_0"])
        self.assertLess(np.linalg.norm(msg_1_bytes_en_0_mx - msg_2_bytes_en_0_mx), _EPSILON)
        self.assertLess(np.linalg.norm(msg_1_bytes_en_0_vec - msg_2_bytes_en_0_vec), _EPSILON)
        self.assertLess(np.linalg.norm(msg_1_bytes_en_0_val - msg_2_bytes_en_0_val), _EPSILON)

        msg_1_bytes_en_1 = msg_body_1["encryption_bytes_1"]
        msg_2_bytes_en_1 = msg_body_2["encryption_bytes_1"]
        for i in range(len(msg_1_bytes_en_1)):
            temp1 = pickle.loads(msg_1_bytes_en_1[i])
            temp2 = pickle.loads(msg_2_bytes_en_1[i])
            msg_1_bytes_en_0_mx = decryptMatrixPaillier(temp1["encrypt_mx_0"], private_key_1)
            msg_1_bytes_en_0_vec = decryptVectorPaillier(temp1["encrypt_vec_0"], private_key_1)
            msg_1_bytes_en_0_val = private_key_1.decrypt(temp1["encrypt_val_0"])
            msg_2_bytes_en_0_mx = decryptMatrixPaillier(temp2["encrypt_mx_0"], private_key_2)
            msg_2_bytes_en_0_vec = decryptVectorPaillier(temp2["encrypt_vec_0"], private_key_2)
            msg_2_bytes_en_0_val = private_key_2.decrypt(temp2["encrypt_val_0"])
            self.assertLess(np.linalg.norm(msg_1_bytes_en_0_mx - msg_2_bytes_en_0_mx), _EPSILON)
            self.assertLess(np.linalg.norm(msg_1_bytes_en_0_vec - msg_2_bytes_en_0_vec), _EPSILON)
            self.assertLess(np.linalg.norm(msg_1_bytes_en_0_val - msg_2_bytes_en_0_val), _EPSILON)
        
        # models
        msg_1_model_bytes_0 = msg_body_1["model_bytes_0"]
        msg_2_model_bytes_0 = msg_body_2["model_bytes_0"]
        for i in range(len(msg_1_model_bytes_0)):
            temp1 = pickle.loads(msg_1_model_bytes_0[i])
            temp2 = pickle.loads(msg_2_model_bytes_0[i])
            msg_1_w = temp1["w"]
            msg_1_b = temp1["b"]
            msg_2_w = temp2["w"]
            msg_2_b = temp2["b"]
            self.assertLess(np.linalg.norm(msg_1_w - msg_2_w), _EPSILON)
            self.assertLess(np.linalg.norm(msg_1_b - msg_2_b), _EPSILON)

        temp1 = pickle.loads(msg_body_1["model_bytes_1"])
        temp2 = pickle.loads(msg_body_2["model_bytes_1"])
        msg_1_w = temp1["w"]
        msg_1_b = temp1["b"]
        msg_2_w = temp2["w"]
        msg_2_b = temp2["b"]
        self.assertLess(np.linalg.norm(msg_1_w - msg_2_w), _EPSILON)
        self.assertLess(np.linalg.norm(msg_1_b - msg_2_b), _EPSILON)

        # strings, ints, bytes
        msg_1_numpy_array_bytes_0 = msg_body_1["numpy_array_bytes_0"]
        msg_2_numpy_array_bytes_0 = msg_body_2["numpy_array_bytes_0"]
        for i in range(len(msg_1_numpy_array_bytes_0)):
            temp1 = bytes_to_numpy_array(msg_1_numpy_array_bytes_0[i])
            temp2 = bytes_to_numpy_array(msg_2_numpy_array_bytes_0[i])
            self.assertLess(np.linalg.norm(temp1 - temp2), _EPSILON)

        temp1 = bytes_to_numpy_array(msg_body_1["numpy_array_bytes_1"])
        temp2 = bytes_to_numpy_array(msg_body_2["numpy_array_bytes_1"])
        self.assertLess(np.linalg.norm(temp1 - temp2), _EPSILON)


if __name__ == '__main__':
    logging.info("Test grpc_client ...")
    unittest.main()
