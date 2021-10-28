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

import numpy as np
# from utils.data_transfer import serialize, deserialize
import requests
import time

class Secure_Layer(object):

    def __init__(self, *args, **kwargs):
        self.config = kwargs
        self.secure_inference_url = args[0]

    def prep(self, x, img_id):
        pass

    def server_call(self, _x):
        # t0 = time.time()
        # msg, _ = serialize({'_x': _x}, **self.config)
        # print('server_call_serialize', time.time() - t0)
        # t0 = time.time()
        # res_msg = requests.post(self.secure_inference_url, data=msg) #
        # print('server_call', time.time() - t0)
        # t0 = time.time()
        # res_dict = deserialize(res_msg.text)
        # print('server_call_deserialize', time.time() - t0)
        # return res_dict
        pass

    def postp(self, msg_dict, img_id):
        pass

    def forward(self,x):
        t0 = time.time()
        _x = self.prep(x)
        print('prep rt: ', time.time()-t0, '_x.shape: ', _x.shape)

        t0 = time.time()
        server_ret_dict = self.server_call(_x)
        print('server rt: ', time.time()-t0)

        t0 = time.time()
        res = self.postp(server_ret_dict )
        print('rt: ', time.time()-t0)
        return res


class Secure_Linear(Secure_Layer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shard = self.config['shard']
        self.p = {}
        self.q = {}
        self.I = {}

    def prep(self, x, img_id):
        # x.shape = (bs, ch, w, h) or (bs, ch)
        if len(x.shape) == 4:
            # for image format:
            # x.shape = bs, ch, w, h
            bs, ch, w, h = x.shape
            self.p[img_id] = np.random.randn(bs, self.shard, 1, 1, 1)
            self.q[img_id] = 100 * np.random.randn(bs, self.shard, ch, w, h)
            self.q[img_id][:, -1:] = - np.sum(self.q[img_id][:, :-1], 1, keepdims=1)
            # m * (x+r) --> m*x + m*r
            _x = np.tile( np.expand_dims(x, 1), (1,self.shard,1,1,1) )
            perturbed_x = self.p[img_id] * ( _x + self.q[img_id] )
            self.I[img_id] = np.reciprocal(self.p[img_id]) / self.shard#np.sum(p, 1, keepdims=1)

        elif len(x.shape) == 2:
            bs, ch = x.shape
            self.p[img_id] = np.random.randn(bs, self.shard, 1)
            self.q[img_id] = 10 * np.random.randn(bs, self.shard, ch)
            self.q[img_id][:, -1:] = - np.sum(self.q[img_id][:, :-1], 1, keepdims=1)

            _x = np.tile( np.expand_dims(x, 1), (1,self.shard,1) )
            perturbed_x = np.tile(self.p[img_id], (1,1,ch)) * ( _x + self.q[img_id] )
            self.I[img_id] = np.reciprocal(self.p[img_id]) / self.shard #np.sum(p, 1, keepdims=1)

        else:
            raise(f"x shape {x.shape} not supported!")

        return perturbed_x

    def postp(self, res_dict, img_id):
        # get input activation
        _z, _b = np.asarray(res_dict['_z']), np.asarray(res_dict['_b'])
        # assemble result 
        if len(_z.shape) == 5:
            bs, shard, ch_out, w_out, h_out = _z.shape
            tmp = _z * np.tile(self.I[img_id], (1,1,ch_out,w_out,h_out)) # element-wise division
            ax_dot_t = np.sum(tmp, axis=1, keepdims=0) # (bs, ch_out, w_out, h_out)
            b_dot_t = _b.reshape(bs, ch_out, 1, 1) # (bs, ch_out, w_out, h_out)
            y_dot_t = ax_dot_t + b_dot_t # (bs, ch_out, w_out, h_out)

        elif len(_z.shape) == 3:
            bs, shard, ch_out = _z.shape
            tmp = _z * np.tile(self.I[img_id], (1,1,ch_out)) # element-wise division
            ax_dot_t = np.sum(tmp, axis=1, keepdims=0) # (bs, ch_out)
            b_dot_t = _b.reshape(bs, ch_out) # (bs, ch_out)
            y_dot_t = ax_dot_t + b_dot_t # (bs, ch_out)

        else:
            print(f"server return format error! return shape {_z.shape}")

        return y_dot_t


class Secure_Nonlinear(Secure_Layer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_p = {}

    def prep(self, x, img_id):
        # perturb activation
        self.p = np.random.randn(*x.shape)
        self.pos_p[img_id] = np.abs(self.p) + 1
        x_prim = x * self.pos_p[img_id]
        return x_prim

    def postp(self, res_dict, img_id):
        z = np.asarray(res_dict['z'])
        # save result
        return z / self.pos_p[img_id]
  
 
class Secure_Res(Secure_Layer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shard = self.config['shard']
        self.p = {}

    def prep(self, x_list, img_id):
        _shape = x_list[0].shape
        self.p[img_id] = np.random.randn(*_shape)
        perturbed_x = np.asarray([self.p[img_id] * xi for xi in x_list])
        return perturbed_x 

    def postp(self, res_dict, img_id):
        z = np.asarray(res_dict['z'])
        return  z / self.p[img_id]

 
class Secure_Reshape(Secure_Layer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shard = self.config['shard']
        self.p = {}

    def prep(self, x, img_id):
        _shape = x.shape
        self.p[img_id] = np.random.randn(*_shape)
        perturbed_x = self.p[img_id] * x
        return perturbed_x 

    def postp(self, res_dict, img_id):
        z = np.asarray(res_dict['y'])
        return  z / self.p[img_id].reshape(*z.shape)

 
# class Secure_Cos(Secure_Layer):
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.shard = self.config['shard']
#
#
#     def do_inference_cos(self):
#         # get input activation
#         input_layer_id = self.compute_graph[self.cur_layer_id]['input']
#         x = self.activation[input_layer_id] # should be a batch of 2 iamges
#         f1, f2 = x[0], x[1]
#         z = f1.dot(f2) / ( np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-5)
#         # call SMPC for secure comparison
#         return z > THRESHOLD


class SP_Client(object):

    def __init__(self, compute_graph, shard=2):
        self.compute_graph = compute_graph
        self.shard = shard
        self.server_base_url = "" # temprary remedy for GRPC purpose
        self.secure_inference_url = self.server_base_url + "secure_inference_orig"
        self.sp_graph = {0: None}  # input layer
        for layer_id, item in self.compute_graph.items():
            config = {'shard': self.shard, "cur_layer_id": layer_id}

            layer_name = item["name"]
            if layer_name.startswith("conv"):
                self.sp_graph[layer_id] = Secure_Linear(self.secure_inference_url, **config)

            elif layer_name.startswith("fc"):
                self.sp_graph[layer_id] = Secure_Linear(self.secure_inference_url, **config)

            elif layer_name.startswith("relu"):
                self.sp_graph[layer_id] = Secure_Nonlinear(self.secure_inference_url, **config)

            elif layer_name.startswith("res"):
                self.sp_graph[layer_id] = Secure_Res(self.secure_inference_url, **config)

            elif layer_name.startswith("flatten"):
                self.sp_graph[layer_id] = Secure_Reshape(self.secure_inference_url, **config)

            # elif layer_name.startswith("cos"):
            #    return self.cos_layer(data_dict)

            else:
                raise NotImplementedError(f"layer not found! layer_id: {layer_id}, layer name: {layer_name}")


    def prep(self, activation, cur_layer_id, img_id):
        input_layer_id = self.compute_graph[cur_layer_id]['input']
        if isinstance(input_layer_id, list):
            x = [activation[i] for i in input_layer_id]
        else:
            x = activation[input_layer_id]

        cur_layer = self.sp_graph[cur_layer_id]
        return cur_layer.prep(x, img_id)

    def postp(self, x, cur_layer_id, img_id):
        cur_layer = self.sp_graph[cur_layer_id]
        return cur_layer.postp(x, img_id)


