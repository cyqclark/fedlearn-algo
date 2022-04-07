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

import torch
import numpy as np
from demos.secure_inference.secure_async.utils.data_transfer import serialize, deserialize

TEST = 1

class Secure_Layer(object):

    def __init__(self, layer_id, layer_name, input_layer, *args, **kwargs):
        self.layer_id = layer_id
        self.input_layer = input_layer
        self.layer_name = layer_name 
        self.kwargs = kwargs
        self.rand_t = {} # for different input id, fixed max size to avoid overflow

    def prep(self, x_dict):
        _x = x_dict['_x']
        _id = x_dict['_id']
        if self.layer_id == 1: # input layer is not affected
            return _x
        print(_x.shape, 'pre prep')

        if isinstance(self.input_layer, list): # res layer
            for i, layer in enumerate(self.input_layer):
                _prev_rand_t = layer.rand_t[_id]
                _shape = [1] * len(_x[i].shape)
                _shape[0] = _x[i].shape[0] # TODO: add more dims aside from batch dim
                print(_x[i].shape,  _prev_rand_t.shape, _shape)
                _x[i] = _x[i] / _prev_rand_t.reshape(*_shape)
        else:
            _shape = [1] * len(_x.shape)
            _shape[0] = _x.shape[0] # TODO: add more dims aside from batch dim
            _prev_rand_t = self.input_layer.rand_t[_id]
            _x = _x / _prev_rand_t.reshape(*_shape)

        print(_x.shape, 'post prep')
        return _x
           
    def postp(self, data_dict):
        pass

    def compute(self, _x):
        pass

    def __call__(self, request_msg):
        x_dict = {
            '_x': deserialize(request_msg.body['data'])['_x'],
            '_id': request_msg.body['_id']
        }
        _x = self.prep(x_dict)
        y_dict = self.compute(_x)
        y_dict.update({'_id': x_dict['_id']})
        z_dict = self.postp(y_dict)
        response_msg_body = {'_id': x_dict['_id'], 'data': serialize(z_dict)}
        return response_msg_body


class Secure_Conv(Secure_Layer):

    def __init__(self, layer_id, layer_name, input_layer, *args, **kwargs):
        super().__init__(layer_id, layer_name, input_layer, *args, **kwargs)

        self.type = 'conv'
        self.weight = torch.tensor(kwargs["weight"])#, device="cuda:0")
        self.bias = kwargs["bias"]
        self.stride = self.kwargs["stride"]
        self.padding = self.kwargs["pad"]

    def compute(self, _x):
        # x.shape = bs, shard, ch_in, w, h
        bs, shard, ch_in, w_in, h_in = _x.shape
        _v = _x.reshape(bs*shard, ch_in, w_in, h_in)
        #tt0=time.time()
        tensor_v = torch.tensor(_v)#, device="cuda:0")
        #print('tensor convert rt: ', time.time()-tt0)

        #t0=time.time()
        av = torch.nn.functional.conv2d(tensor_v, self.weight, bias=None, stride=self.stride, padding=self.padding)
        #print('torch core rt: ', time.time()-t0)
        av = av.cpu().numpy() # (bs*shard, ch_out, w_out, h_out)
        _, ch_out, w_out, h_out = av.shape
        av = av.reshape(bs, shard, ch_out, w_out, h_out)

        return {'av' : av}

    def postp(self, y_dict):
        # rand_t.shape = (bs,)
        _id = y_dict['_id']

        bs, shard, ch_out, w_out, h_out = y_dict['av'].shape

        if _id not in self.rand_t:
            _shape = list(y_dict['av'].shape)
            for i in range(1, len(_shape), 1):
                _shape[i] = 1
            self.rand_t[_id] = np.random.rand(bs).astype('float32')

        if TEST:  self.rand_t[_id] = np.ones_like(self.rand_t[_id])

        rand_t_for_ax = self.rand_t[_id].reshape(bs, 1, 1, 1, 1)
        _z =  y_dict['av'] * rand_t_for_ax 
        rand_t_for_bias = self.rand_t[_id].reshape(bs, 1)
        _b = self.bias.reshape(1, ch_out) * rand_t_for_bias 
        return {'_z': _z, '_b':_b}




class Secure_Linear(Secure_Layer):

    def __init__(self, layer_id, layer_name, input_layer, *args, **kwargs):
        super().__init__(layer_id, layer_name, input_layer, *args, **kwargs)
        self.type = 'linear'
        self.weight = torch.tensor(kwargs["weight"])#, device="cuda:0")
        self.bias = kwargs["bias"]


    def compute(self, _x):
        bs, shard, ch_in = _x.shape
        _v = _x.reshape(bs*shard, ch_in)

        tensor_v = torch.tensor(_v)#, device="cuda:0")
        av = torch.nn.functional.linear(tensor_v, self.weight, bias=None)
        av = av.cpu().numpy() # (bs*shard, ch_out)
        _, ch_out = av.shape
        av = av.reshape(bs, shard, ch_out)
        return {"av": av}

    def postp(self, y_dict):
        # rand_t.shape = (bs,)
        _id = y_dict['_id']

        bs, shard, ch_out = y_dict['av'].shape
        if _id not in self.rand_t:
            self.rand_t[_id] = np.random.rand(bs).astype('float32')
        if TEST:  self.rand_t[_id] = np.ones_like(self.rand_t[_id])

        rand_t_for_ax = self.rand_t[_id] .reshape(bs, 1, 1)
        _z =  y_dict['av'] * rand_t_for_ax 
        rand_t_for_bias = self.rand_t[_id] .reshape(bs, 1 )
        _b = self.bias.reshape(1, ch_out)* rand_t_for_bias 
        return {'_z': _z, '_b': _b}


class Secure_Flatten(Secure_Layer):

    def __init__(self, layer_id, layer_name, input_layer, *args, **kwargs):
        super().__init__(layer_id, layer_name, input_layer, *args, **kwargs)
        self.type = 'flatten'

#    def prep(self, _x):
#        layer_id = data_dict['cur_layer_id']
#
#        print('xshape', _x.shape)
#        bs, ch, w, h = _x.shape
#        prev_rand_t = self.rand_t[layer_id-1]
#        input_type = self.compute_graph[layer_id-1]['_type']
#        if input_type == 'RES':
#            prev_rand_t = prev_rand_t[:bs]
#        _x = _x / prev_rand_t

    def compute(self, x):
        return {'x': x}

    def postp(self, y_dict):
        # rand_t.shape = (bs,)
        _id = y_dict['_id']

        if _id not in self.rand_t:
            bs = y_dict['x'].shape[0]
            self.rand_t[_id] = np.random.rand(bs, 1).astype('float32')
        if TEST:  self.rand_t[_id] = np.ones_like(self.rand_t[_id])

        bs, ch, w, h = y_dict['x'].shape
        y = y_dict['x'].reshape((bs, ch*w*h)) * self.rand_t[_id]
        return {'y': y}


class Secure_ReLU(Secure_Layer):

    def __init__(self, layer_id, layer_name, input_layer, *args, **kwargs):
        super().__init__(layer_id, layer_name, input_layer, *args, **kwargs)
        self.type = 'relu'
        self.p = torch.tensor(kwargs["weight"])

    def compute(self, _x):
        #if self.input_layer.type in ['conv', 'linear']: 
        print(_x.shape, self.p.shape)
        tensor_x = torch.tensor(_x)
        y = torch.nn.functional.prelu(tensor_x, self.p)
        y = y.cpu().numpy()
        return {'y' : y}

    def postp(self, y_dict):
        # rand_t.shape = input.shape
        _id = y_dict['_id']

        if _id not in self.rand_t:
            _shape = list(y_dict['y'].shape)
            for i in range(1, len(_shape), 1):
                _shape[i] = 1
            self.rand_t[_id] = np.random.rand(*_shape).astype('float32')
        if TEST:  self.rand_t[_id] = np.ones_like(self.rand_t[_id])
        #bs, ch_in, w_in, h_in = _x.shape # consider for 2D inputs as well
        z = y_dict['y'] * self.rand_t[_id]
        return {'z': z}


class Secure_Res(Secure_Layer):

    def __init__(self, layer_id, layer_name, input_layer, *args, **kwargs):
        super().__init__(layer_id, layer_name, input_layer, *args, **kwargs)
        self.type = 'res'

    def compute(self, _x):
        return {"y": np.sum(_x, 0, keepdims=0)}

    def postp(self, y_dict):
        # rand_t.shape = (bs,)
        _id = y_dict['_id']

        if _id not in self.rand_t:
            bs = y_dict['y'].shape[0]
            self.rand_t[_id] = np.random.rand(bs).astype('float32')
        if TEST:  self.rand_t[_id] = np.ones_like(self.rand_t[_id])

        _shape = [1] * len(y_dict['y'].shape)
        _shape[0] = y_dict['y'].shape[0]
        z = y_dict["y"] * self.rand_t[_id].reshape(*_shape)
        return {'z': z}

class SP_Server(object):

    def __init__(self, compute_graph, parameters):
        self.compute_graph = compute_graph 
        self.parameters = parameters
        self.sp_graph = {0: None} # input layer

        for layer_id, item in self.compute_graph.items():
            if isinstance(item["input"], list):
                input_layer = [ self.sp_graph[i] for i in item["input"] ]
            else:
                input_layer = self.sp_graph[item["input"]]

            layer_name = item["name"]
            if layer_name.startswith("conv"): 
                self.sp_graph[layer_id] = Secure_Conv(layer_id, layer_name, input_layer, **self.parameters[layer_name])
    
            elif layer_name.startswith("fc"):
                self.sp_graph[layer_id] = Secure_Linear(layer_id, layer_name, input_layer, **self.parameters[layer_name])
    
            elif layer_name.startswith("relu"):
                self.sp_graph[layer_id] = Secure_ReLU(layer_id, layer_name, input_layer, **self.parameters[layer_name])
    
            elif layer_name.startswith("res"):
                self.sp_graph[layer_id] = Secure_Res(layer_id, layer_name, input_layer, **self.parameters[layer_name])
    
            elif layer_name.startswith("flatten"):
                self.sp_graph[layer_id] = Secure_Flatten(layer_id, layer_name, input_layer)
    
            #elif layer_name.startswith("cos"):
            #    return self.cos_layer(data_dict)

            else:
                raise NotImplementedError(f"layer not found! layer_id: {layer_id}, layer name: {layer_name}")



    def forward(self, data_dict, **kwargs):
        print('layer_id', data_dict['cur_layer_id'], data_dict.keys())

        cur_layer = self.sp_graph[data_dict['cur_layer_id']]
        _x = data_dict['_x'].astype('float32')
        res = cur_layer(_x)
        return res
   

    def get_layer(self, cur_layer):
        cur_layer = self.sp_graph[cur_layer]
        return cur_layer
