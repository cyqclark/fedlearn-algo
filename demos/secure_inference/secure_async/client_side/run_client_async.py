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
from core.entity.common.message import RequestMessage, ResponseMessage
from core.entity.common.machineinfo import MachineInfo
from demos.secure_inference.secure_async.utils.grpc_async_communicator import AsyncGRPCCommunicator
import socket
import cv2
from demos.secure_inference.secure_async.utils.data_transfer import serialize, deserialize
from core.grpc_comm.grpc_node import send_request
import json
from client_protocol import SP_Client
import uuid
from collections import  defaultdict

# global variables
_EPSILON = 1e-8
try:
    _LOCALHOST = socket.gethostbyname(socket.gethostname())
except socket.gaierror:
    # if failed to fetch 127.0.0.1, try 0.0.0.0
    _LOCALHOST = socket.gethostbyname("")


THRESHOLD = 0.295

def get_cosdist(f1: np.ndarray, f2: np.ndarray):
    if isinstance(f1, list):
        f1 = np.asarray(f1)
    if isinstance(f2, list):
        f2 = np.asarray(f2)
    return f1.dot(f2) / ( np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-5)


class Client(object):
    def __init__(self, client_info, server_info):
        #
        self.client_info = client_info
        self.server_info = server_info
        self.machine_info = self.client_info
        #
        self.sps = self.get_comput_graph()
        self.dict_functions = {}
        self.function_register()
        self.activation = defaultdict(dict)

    def get_comput_graph(self):
        request = RequestMessage(sender = self.machine_info,
                                 receiver = self.server_info,
                                 body = {"haha": 1},  # numpy tensor
                                 phase_id = 'init_comput_graph')
        response = send_request(request)
        self.compute_graph_server = json.loads(response.body['graph'])

        self.compute_graph_server = {int(k): v for k, v in self.compute_graph_server.items()}  # bug in orjson
        self.last_layer_id = max(self.compute_graph_server.keys())
        return SP_Client(self.compute_graph_server, shard=2)

    def function_register(self): # all layers on the client side shares the same formatting
        for i in range(51):
            self.dict_functions[f"layer_{i}"] = self.forward_single_layer
        self.dict_functions["post_inference_session"] = self.post_inference_session

    def forward_single_layer(self, msg):
        phase_id = msg.phase_id
        img_id = msg.body['_id']
        cur_layer_id = int(phase_id.split('_')[1])

        y = deserialize(msg.body['data'])  # parse from server return
        # for k, v in y.items():
        #     print(f'pre postp for layer {cur_layer_id} is: ', k, v.flatten()[:10])
        self.activation[img_id][cur_layer_id] = self.sps.postp(y, cur_layer_id, img_id)
        # print(f'activation for layer {cur_layer_id} is: ', self.activation[img_id][cur_layer_id].flatten()[:10])
        if cur_layer_id == 50: # last layer
            _x = self.activation[img_id][cur_layer_id]
            next_phase_id = "post_inference_session"
            _sender, _reciever = self.server_info, self.server_info
        else:
            _x = self.sps.prep(self.activation[img_id], cur_layer_id+1, img_id)
            next_phase_id = f"layer_{cur_layer_id + 1}"  # this needs to be align with the dict_functions in server_SecInf
            _sender, _reciever = self.server_info, self.client_info

        response = ResponseMessage(sender = _sender,
                                   receiver = _reciever,
                                   body={'_id': img_id, 'data': serialize({'_x': _x})},  # numpy tensor
                                   phase_id=next_phase_id)
        return response

    def process_queue(self, request):
        phase_id = request.phase_id
        cur_layer_id = int(phase_id.split('_')[1])
        if cur_layer_id < 50:
            next_phase_id = f"layer_{cur_layer_id+1}" # this needs to be align with the dict_functions in server_SecInf
        else:
            next_phase_id = "post_inference_session"

        if next_phase_id not in self.dict_functions.keys():
            raise ValueError("%s, Function %s is not implemented." % (self.__class__.__name__, next_phase_id))
        response = self.dict_functions[next_phase_id](request)
        return response

    def get_init_req(self, data_dict):
        self.activation[data_dict['_id']] = {0: data_dict['img']}

        _x = self.sps.prep(self.activation[data_dict['_id']], 1, data_dict['_id'])
        init_req = RequestMessage(sender=self.client_info,
                                  receiver=self.server_info,
                                  body={'_id': data_dict['_id'], 'data': serialize({'_x': _x})},  # numpy tensor
                                  phase_id="layer_1")
        return init_req

    def post_inference_session(self, msg): # TODO: put to server side
        img_id = msg.body['_id']
        y = deserialize(msg.body['data'])  # parse from server return
        self.activation[img_id][self.last_layer_id] = self.sps.postp(y, self.last_layer_id, img_id)
        self.last_activation = self.activation[img_id][self.last_layer_id]
        cosdist = get_cosdist( self.last_activation[0], self.last_activation[1] )
        res = {'feature': self.last_activation, 'dist': cosdist, 'pred': int(cosdist > THRESHOLD), '_id': img_id}

        response = ResponseMessage(sender = self.server_info,
                                   receiver = self.client_info,
                                   body= res,  # numpy tensor
                                   phase_id='finish')
        return response

    def get_input(self, n=1):

        with open('../../data/FaceRecognition/LFW/pairs.txt') as f:
            pairs_lines = f.readlines()[290:]

        for i in range(n):
            p = pairs_lines[i].replace('\n','').split('\t')

            if 3==len(p):
                sameflag = 1
                name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
                name2 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[2]))
            if 4==len(p):
                sameflag = 0
                name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
                name2 = p[2]+'/'+p[2]+'_'+'{:04}.jpg'.format(int(p[3]))

            img1 = cv2.imread("../../data/FaceRecognition/LFW/lfw_processed/"+name1)
            img2 = cv2.imread("../../data/FaceRecognition/LFW/lfw_processed/"+name2)
            img1_normalized = (img1.transpose(2, 0, 1)-127.5)/128.0
            img2_normalized = (img2.transpose(2, 0, 1)-127.5)/128.0

            yield str(uuid.uuid4()), np.stack([img1_normalized, img2_normalized], 0).astype('float32'), sameflag

if __name__ == '__main__':
    import time
    # machine information
    server_info = MachineInfo(ip=_LOCALHOST, port='38892', token='simulated master for test')
    client_info = MachineInfo(ip=_LOCALHOST, port='38893', token='simulated client for test')
    client = Client(client_info, server_info)
    client_async = AsyncGRPCCommunicator(client)
    client_async.start_grpc_message_processing()

    t0 = time.time()
    print( 'current time: ', time.time())
    for _id, img, label in client.get_input(20):
        print('sending image id:', _id, " time stamp: ", time.time())
        init_req = client.get_init_req({'_id': _id, 'img': img})
        client_async.send_message(init_req)
    # stop
    # client_async.stop_grpc_node_receive_routine()

    print( 'runtime: ', time.time() - t0)
    s_seconds = 2000
    print("after a to b, sleep %s seconds." % s_seconds)
    time.sleep(s_seconds)
