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
# from core.client.client import Client
from core.server.server import Server
from core.entity.common.message import RequestMessage
from core.entity.common.machineinfo import MachineInfo
from protocol_client import SP_Client

import numpy as np
import cv2
from core.grpc_comm.grpc_node import send_request
import json
from demos.secure_inference.secure_sync.utils.data_transfer import serialize, deserialize
import time
import torch

THRESHOLD = 0.295

def get_cosdist(f1: np.ndarray, f2: np.ndarray):
    if isinstance(f1, list):
        f1 = np.asarray(f1)
    if isinstance(f2, list):
        f2 = np.asarray(f2)
    print(f1.shape, f2.shape)
    return f1.dot(f2) / ( np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-5)



class Insecure_Client(object):

    def __init__(self):
        self.torch_model = sphere20a(feature=True).cpu()
        pretrained_weights = torch.load('../../data/FaceRecognition/sphere20a_20171020.pth')
        pretrained_weights_for_inference = {k:v for k, v in pretrained_weights.items() if 'fc6' not in k}
        self.torch_model.load_state_dict(pretrained_weights_for_inference )

    def inference(self, raw_img):
        t0 = time.time()
        x = torch.tensor(raw_img).cpu()
        _prob = self.torch_model(x).detach().numpy()
        cosdist = get_cosdist(_prob[0], _prob[1])
        return {'feature': _prob, 'dist': cosdist, 'pred': int(cosdist > THRESHOLD), 'runtime': time.time()-t0}

class Secure_Client(Server):
    def __init__(self, server_info, machine_info: MachineInfo):
        super().__init__()
        self.algorithm_type = 'kernelmethod'
        self.source_data = None
        self.response = None
        self.machine_info = machine_info
        self.server_info = server_info
        self.model_param = None
        self.kernel_mapping_param = None
        self.kernel_mapping = None
        self.feature_dim = 0
        self.map_dim = 0
        self.sample_num = 0
        self.data = None
        self.is_active = False
        self.data_path = None
        self.model_path = None
        self.model = {}
        self.dict_functions = {}
        self.shard = 2
        self.activation = {} # key: layer, value: activation


        # TODO: replace with RPC call
        # res = requests.get(self.server_base_url + "get_compute_graph")
        # self.compute_graph_server = orjson.loads(res.text)
        request = RequestMessage(sender = self.machine_info,
                                 receiver = self.server_info,
                                 body = {"haha": 1},  # numpy tensor
                                 phase_id = 'init_comput_graph')
        # response = call_grpc_client(request)
        response = send_request(request)
        self.compute_graph_server = json.loads(response.body['graph'])

        self.compute_graph_server = {int(k): v for k, v in self.compute_graph_server.items()}  # bug in orjson
        self.last_layer_id = max(self.compute_graph_server.keys())
        self.sps = SP_Client(self.compute_graph_server, shard=2)

    def get_next_phase(self, phase: str) -> str:
        """
        Transfer old phase of client to new phase of server
        """
        # inference
        if phase == "inference_init":
            next_phase = "layer_1"
        elif phase.startswith("layer_"):
            layer_id = int(phase.split('_')[1])
            if layer_id < 50:
                next_phase = f"layer_{layer_id+1}"
            else:
                next_phase = "inference_end"
        else:
            raise ValueError("Cannot find phase %s in both train and inference!"%phase)
        return next_phase

    def init_training_control(self):
        pass

    def init_inference_control(self):
        pass

    def is_training_continue(self):
        pass

    def post_training_session(self):
        pass

    def is_inference_continue(self, cur_layer_id):
        if cur_layer_id == 0:
            return True
        if self.compute_graph_server[cur_layer_id]["is_stop"] == 1:
            self.last_activation = self.sps.activation[cur_layer_id]
            return False
        else:
            return True


    def secure_inference(self, x, layer_id=1, is_parallel=True) -> None:

        """
        init_phase can be skiped, clients is actually the server providing model weights
        """
        t0 = time.time()

        phase_id = f"layer_{layer_id}" # this needs to be align with the dict_functions in server_SecInf
        self.sps.activation[0] = x

        while True:
            _x = self.sps.prep(layer_id)

            # RPC call
            request = RequestMessage(sender = self.machine_info,
                                     receiver = self.server_info,
                                     body = {'data' : serialize({'_x': _x})},  # numpy tensor
                                     phase_id = phase_id)
            # response = call_grpc_client(request, is_parallel)
            response = send_request(request)
            y = deserialize(response.body['data'])  # parse from server return

            self.sps.postp(y, layer_id)
            if not self.is_inference_continue(layer_id):
                break
            phase_id = self.get_next_phase(phase_id)
            layer_id = int(phase_id.split('_')[1])


        res = self.post_inference_session()
        res.update( {'runtime': time.time() - t0} )
        return res

    def post_inference_session(self):
        self.last_activation = self.sps.activation[self.last_layer_id]
        cosdist = get_cosdist( self.last_activation[0], self.last_activation[1] )
        response = {'feature': self.last_activation, 'dist': cosdist, 'pred': int(cosdist > THRESHOLD)}
        return response



def get_input(n=1000):

    with open('../../data/FaceRecognition/LFW/pairs.txt') as f:
        pairs_lines = f.readlines()[1:]

    img_label = []
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

        img_label.append( [np.stack([img1_normalized, img2_normalized], 0).astype('float32'), sameflag] )
    return img_label



if __name__ == '__main__':
    import socket
    from demos.secure_inference.insecure.model_sphereface import sphere20a

    try:
        _LOCALHOST = socket.gethostbyname(socket.gethostname())
    except socket.gaierror:
        # if failed to fetch 127.0.0.1, try 0.0.0.0
        _LOCALHOST = socket.gethostbyname("")

    # machine information
    master_info = MachineInfo(ip="127.0.0.1", port='8890', token='master_machine')
    machine_info = MachineInfo(ip="127.0.0.1", port='8891', token='client_1')

    print(master_info)
    print(machine_info)

    secure_client = Secure_Client(master_info, machine_info)
    insecure_client = Insecure_Client()
    raw_img_set = get_input()  #
    correct = 0
    for i, (raw_img, sameflag) in enumerate(raw_img_set):
        ref = insecure_client.inference(raw_img)
        res = secure_client.secure_inference(raw_img)
        correct += int(ref['pred'] == res['pred'])
        print("label: %r;   Pred_comp: %r / %r;   Match: %r/%r;   Time_comp %.2fs / %.2fs;   Dist_comp: %.6f / %.6f" % ( sameflag,  ref['pred'], res['pred'], correct, (i+1), ref['runtime'], res['runtime'], ref['dist'], res['dist'] ) )
