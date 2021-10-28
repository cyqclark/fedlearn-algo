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

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from core.entity.common.message import ResponseMessage
from core.entity.common.machineinfo import MachineInfo
from core.client.client import Client

from demos.secure_inference.insecure.model_sphereface import sphere20a
from demos.secure_inference.secure_async.server_side.server_protocol import SP_Server
from typing import Set
import torch
import json

def check_message(msg: str, target: Set):
    if msg in target:
        return True
    else:
        return False


class Secure_Server(Client):
    def __init__(self, machine_info: MachineInfo):
        super().__init__()
        self.machine_info = machine_info
        self.dict_functions = {}

        torch_model = sphere20a(feature=True)
        pretrained_weights = torch.load('../../data/FaceRecognition/sphere20a_20171020.pth')  # TODO: make it as input config
        pretrained_weights_for_inference = {k: v for k, v in pretrained_weights.items() if 'fc6' not in k}
        torch_model.load_state_dict(pretrained_weights_for_inference)
        self.compute_graph, self.parameters = self.load_from_torch_model(torch_model)
        self.rand_t = {0: 1}  # no randomization before the first layer
        self.sps = SP_Server(self.compute_graph, self.parameters)

        self.function_registration()

        self.grpc_node.set_machine_info(self.machine_info)
        self.grpc_servicer.set_msg_processor(self)

    def set_msg_processor(self):
        pass

    def metric(self) -> None:
        return

    def init_comput_graph(self, request):
        response = ResponseMessage(sender = request.client_info,
                                   receiver = request.server_info,
                                   body = {'graph': json.dumps(self.compute_graph)},
                                   phase_id = 'init_end')
        return response

    def load_from_torch_model(self, torch_model):
        # TODO: auto derive from pytorch graph definition, do a single forward pass and record the execution order
        compute_graph = {  # this will be sent to client
            ##################### block1 ###################
            1: {"name": "conv1_1", "input": 0, "_type": "LINEAR", "port": 5001, "is_stop": 0},
            2: {"name": "relu1_1", "input": 1, "_type": "NONLINEAR", "port": 5001, "is_stop": 0},
            3: {"name": "conv1_2", "input": 2, "_type": "LINEAR", "port": 5001, "is_stop": 0},
            4: {"name": "relu1_2", "input": 3, "_type": "NONLINEAR", "port": 5001, "is_stop": 0},
            5: {"name": "conv1_3", "input": 4, "_type": "LINEAR", "port": 5001, "is_stop": 0},
            6: {"name": "relu1_3", "input": 5, "_type": "NONLINEAR", "port": 5001, "is_stop": 0},
            7: {"name": "res1_3", "input": [2, 6], "_type": "RES", "port": 5001, "is_stop": 0},

            #################### block2 ###################
            # first part
            8: {"name": "conv2_1", "input": 7, "_type": "LINEAR", "port": 5001, "is_stop": 0},
            9: {"name": "relu2_1", "input": 8, "_type": "NONLINEAR", "port": 5001, "is_stop": 0},
            10: {"name": "conv2_2", "input": 9, "_type": "LINEAR", "port": 5001, "is_stop": 0},
            11: {"name": "relu2_2", "input": 10, "_type": "NONLINEAR", "port": 5001, "is_stop": 0},
            12: {"name": "conv2_3", "input": 11, "_type": "LINEAR", "port": 5001, "is_stop": 0},
            13: {"name": "relu2_3", "input": 12, "_type": "NONLINEAR", "port": 5001, "is_stop": 0},
            14: {"name": "res2_3", "input": [9, 13], "_type": "RES", "port": 5001, "is_stop": 0},
            # second part
            15: {"name": "conv2_4", "input": 14, "_type": "LINEAR", "port": 5001, "is_stop": 0},
            16: {"name": "relu2_4", "input": 15, "_type": "NONLINEAR", "port": 5001, "is_stop": 0},
            17: {"name": "conv2_5", "input": 16, "_type": "LINEAR", "port": 5001, "is_stop": 0},
            18: {"name": "relu2_5", "input": 17, "_type": "NONLINEAR", "port": 5001, "is_stop": 0},
            19: {"name": "res2_5", "input": [14, 18], "_type": "RES", "port": 5001, "is_stop": 0},

            #################### block3 ###################
            # first part
            20: {"name": "conv3_1", "input": 19, "_type": "LINEAR", "port": 5001, "is_stop": 0},
            21: {"name": "relu3_1", "input": 20, "_type": "NONLINEAR", "port": 5001, "is_stop": 0},
            22: {"name": "conv3_2", "input": 21, "_type": "LINEAR", "port": 5001, "is_stop": 0},
            23: {"name": "relu3_2", "input": 22, "_type": "NONLINEAR", "port": 5001, "is_stop": 0},
            24: {"name": "conv3_3", "input": 23, "_type": "LINEAR", "port": 5001, "is_stop": 0},
            25: {"name": "relu3_3", "input": 24, "_type": "NONLINEAR", "port": 5001, "is_stop": 0},
            26: {"name": "res3_3", "input": [21, 25], "_type": "RES", "port": 5001, "is_stop": 0},
            # second part
            27: {"name": "conv3_4", "input": 26, "_type": "LINEAR", "port": 5001, "is_stop": 0},
            28: {"name": "relu3_4", "input": 27, "_type": "NONLINEAR", "port": 5001, "is_stop": 0},
            29: {"name": "conv3_5", "input": 28, "_type": "LINEAR", "port": 5001, "is_stop": 0},
            30: {"name": "relu3_5", "input": 29, "_type": "NONLINEAR", "port": 5001, "is_stop": 0},
            31: {"name": "res3_5", "input": [26, 30], "_type": "RES", "port": 5001, "is_stop": 0},
            # third part
            32: {"name": "conv3_6", "input": 31, "_type": "LINEAR", "port": 5001, "is_stop": 0},
            33: {"name": "relu3_6", "input": 32, "_type": "NONLINEAR", "port": 5001, "is_stop": 0},
            34: {"name": "conv3_7", "input": 33, "_type": "LINEAR", "port": 5001, "is_stop": 0},
            35: {"name": "relu3_7", "input": 34, "_type": "NONLINEAR", "port": 5001, "is_stop": 0},
            36: {"name": "res3_7", "input": [31, 35], "_type": "RES", "port": 5001, "is_stop": 0},
            # fourth part
            37: {"name": "conv3_8", "input": 36, "_type": "LINEAR", "port": 5001, "is_stop": 0},
            38: {"name": "relu3_8", "input": 37, "_type": "NONLINEAR", "port": 5001, "is_stop": 0},
            39: {"name": "conv3_9", "input": 38, "_type": "LINEAR", "port": 5001, "is_stop": 0},
            40: {"name": "relu3_9", "input": 39, "_type": "NONLINEAR", "port": 5001, "is_stop": 0},
            41: {"name": "res3_9", "input": [36, 40], "_type": "RES", "port": 5001, "is_stop": 0},

            ##################### block4 ###################
            42: {"name": "conv4_1", "input": 41, "_type": "LINEAR", "port": 5001, "is_stop": 0},
            43: {"name": "relu4_1", "input": 42, "_type": "NONLINEAR", "port": 5001, "is_stop": 0},
            44: {"name": "conv4_2", "input": 43, "_type": "LINEAR", "port": 5001, "is_stop": 0},
            45: {"name": "relu4_2", "input": 44, "_type": "NONLINEAR", "port": 5001, "is_stop": 0},
            46: {"name": "conv4_3", "input": 45, "_type": "LINEAR", "port": 5001, "is_stop": 0},
            47: {"name": "relu4_3", "input": 46, "_type": "NONLINEAR", "port": 5001, "is_stop": 0},
            48: {"name": "res4_3", "input": [43, 47], "_type": "RES", "port": 5001, "is_stop": 0},

            ##################### Linear ###################
            # header
            49: {"name": "flatten", "input": 48, "_type": "RESHAPE", "port": 5001, "is_stop": 0},
            50: {"name": "fc5", "input": 49, "_type": "LINEAR", "port": 5001, "is_stop": 1},
        }

        # weights are stripped off from inference map, won't be sent to client
        parameters = {}
        for name, layer in torch_model.named_modules():
            if name.startswith("conv"):
                parameters[name] = {
                    "weight": layer.weight.detach().numpy(),
                    "bias": layer.bias.detach().numpy(),
                    "stride": layer.stride,
                    "pad": layer.padding,
                }
            elif name.startswith("relu"):
                parameters[name] = {
                    "weight": layer.weight.detach().numpy(),
                }
            elif name.startswith("fc"):
                parameters[name] = {
                    "weight": layer.weight.detach().numpy(),
                    "bias": layer.bias.detach().numpy(),
                }
        for name in ['res1_3', 'res2_3', 'res2_5', 'res3_3', 'res3_5', 'res3_7', 'res3_9', 'res4_3']:
            parameters[name] = {
                "weight": [1, 1],
            }

        return compute_graph, parameters


    def function_registration(self):
        """
        Define the correspondence between response message type and server processing function.
        """
        # inference
        self.dict_functions["init_comput_graph"] = self.init_comput_graph
        for i in range(51):
            self.dict_functions[f"layer_{i}"] = self.sps.get_layer(i)



if __name__ == '__main__':
    import argparse
    import socket

    try:
        _LOCALHOST = socket.gethostbyname(socket.gethostname())
    except socket.gaierror:
        # if failed to fetch 127.0.0.1, try 0.0.0.0
        _LOCALHOST = socket.gethostbyname("")

    parser = argparse.ArgumentParser()
    # machine info
    parser.add_argument('-I', '--ip', type=str, help="string of server ip, exp:%s" % _LOCALHOST)
    parser.add_argument('-P', '--port', type=str, help="string of server port, exp: 8890")
    parser.add_argument('-T', '--token', type=str, help="string of server name, exp: client_1")

    args = parser.parse_args()
    print("ip: %s" % args.ip)
    print("port: %s" % args.port)
    print("token: %s" % args.token)

    # create client object
    server_info = MachineInfo(ip=args.ip, port=args.port, token=args.token)
    server = Secure_Server(server_info)
    server.start_serve_termination_block()

