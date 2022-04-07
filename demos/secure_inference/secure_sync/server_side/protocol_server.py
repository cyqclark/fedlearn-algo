import torch
import numpy as np
from demos.secure_inference.secure_sync.utils.data_transfer import serialize, deserialize
from core.entity.common.message import ResponseMessage

class Secure_Layer(object):

    def __init__(self, layer_id, layer_name, input_layer, *args, **kwargs):
        self.layer_id = layer_id
        self.input_layer = input_layer
        self.layer_name = layer_name 
        self.kwargs = kwargs
        self.rand_t = None

    def prep(self, _x):

        if self.layer_id == 1: # input layer is not affected
            return _x
        print(_x.shape, 'pre prep')

        if isinstance(self.input_layer, list): # res layer
            for i, layer in enumerate(self.input_layer):
                _prev_rand_t = layer.rand_t
                _shape = [1] * len(_x[i].shape)
                _shape[0] = _x[i].shape[0] # TODO: add more dims aside from batch dim
                print(_x[i].shape,  _prev_rand_t.shape, _shape)
                _x[i] = _x[i] / _prev_rand_t.reshape(*_shape)
        else:
            _shape = [1] * len(_x.shape)
            _shape[0] = _x.shape[0] # TODO: add more dims aside from batch dim
            _prev_rand_t = self.input_layer.rand_t
            _x = _x / _prev_rand_t.reshape(*_shape)

        print(_x.shape, 'post prep')
        return _x
           
    def postp(self, data_dict):
        pass

    def compute(self, _x):
        pass

    def __call__(self, request_msg):
        _x = deserialize(request_msg.body['data'])['_x']
        _x = self.prep(_x)
        y_dict = self.compute(_x)
        z_dict = self.postp(y_dict)
        response_msg = ResponseMessage(sender = request_msg.client_info,
                                   receiver = request_msg.server_info,
                                   body = {'data': serialize(z_dict)},
                                   phase_id = request_msg.phase_id + '_done')
        return response_msg


class Secure_Conv(Secure_Layer):

    def __init__(self, layer_id, layer_name, input_layer, *args, **kwargs):
        super().__init__(layer_id, layer_name, input_layer, *args, **kwargs)

        self.type = 'conv'
        self.weight = torch.tensor(kwargs["weight"])
        self.bias = kwargs["bias"]
        self.stride = self.kwargs["stride"]
        self.padding = self.kwargs["pad"]

    def compute(self, _x):
        # x.shape = bs, shard, ch_in, w, h
        bs, shard, ch_in, w_in, h_in = _x.shape
        _v = _x.reshape(bs*shard, ch_in, w_in, h_in)
        #tt0=time.time()
        tensor_v = torch.tensor(_v)
        #print('tensor convert rt: ', time.time()-tt0)

        #t0=time.time()
        av = torch.nn.functional.conv2d(tensor_v, self.weight, bias=None, stride=self.stride, padding=self.padding)
        #print('torch core rt: ', time.time()-t0)
        av = av.cpu().numpy() # (bs*shard, ch_out, w_out, h_out)
        _, ch_out, w_out, h_out = av.shape
        av = av.reshape(bs, shard, ch_out, w_out, h_out)

        return {'av' : av}

    def postp(self, y_dict):
        # post processing

        bs, shard, ch_out, w_out, h_out = y_dict['av'].shape
        if self.rand_t is None:
            _shape = list(y_dict['av'].shape)
            for i in range(1, len(_shape), 1):
                _shape[i] = 1
            self.rand_t = np.random.rand(bs).astype('float32')
        else:
            print('--------self.rand_t is not None! ------')

        rand_t_for_ax = self.rand_t.reshape(bs, 1, 1, 1, 1)
        _z =  y_dict['av'] * rand_t_for_ax 
        rand_t_for_bias = self.rand_t.reshape(bs, 1)
        _b = self.bias.reshape(1, ch_out) * rand_t_for_bias 
        return {'_z': _z, '_b':_b}


class Secure_Linear(Secure_Layer):

    def __init__(self, layer_id, layer_name, input_layer, *args, **kwargs):
        super().__init__(layer_id, layer_name, input_layer, *args, **kwargs)
        self.type = 'linear'
        self.weight = torch.tensor(kwargs["weight"])
        self.bias = kwargs["bias"]


    def compute(self, _x):
        bs, shard, ch_in = _x.shape
        _v = _x.reshape(bs*shard, ch_in)

        tensor_v = torch.tensor(_v)
        av = torch.nn.functional.linear(tensor_v, self.weight, bias=None)
        av = av.cpu().numpy() # (bs*shard, ch_out)
        _, ch_out = av.shape
        av = av.reshape(bs, shard, ch_out)
        return {"av": av}

    def postp(self, y_dict):
        # post processing

        bs, shard, ch_out = y_dict['av'].shape
        if self.rand_t is None:
            self.rand_t = np.random.rand(bs).astype('float32')

        rand_t_for_ax = self.rand_t.reshape(bs, 1, 1)
        _z =  y_dict['av'] * rand_t_for_ax 
        rand_t_for_bias = self.rand_t.reshape(bs, 1 )
        _b = self.bias.reshape(1, ch_out)* rand_t_for_bias 
        return {'_z': _z, '_b': _b} # TODO: double check serialization



class Secure_Flatten(Secure_Layer):

    def __init__(self, layer_id, layer_name, input_layer, *args, **kwargs):
        super().__init__(layer_id, layer_name, input_layer, *args, **kwargs)
        self.type = 'flatten'

    def compute(self, x):
        return {'x': x}

    def postp(self, y_dict):
        # post processing

        if self.rand_t is None:
            bs = y_dict['x'].shape[0]
            self.rand_t = np.random.rand(bs, 1).astype('float32')

        bs, ch, w, h = y_dict['x'].shape
        y = y_dict['x'].reshape((bs, ch*w*h)) * self.rand_t
        return {'y': y}


class Secure_ReLU(Secure_Layer):

    def __init__(self, layer_id, layer_name, input_layer, *args, **kwargs):
        super().__init__(layer_id, layer_name, input_layer, *args, **kwargs)
        self.type = 'relu'
        self.p = torch.tensor(kwargs["weight"])

    def compute(self, _x):
        print("_x.shape=" + str(_x.shape))
        print("p.shape=" + str(self.p.shape))
        tensor_x = torch.tensor(_x)
        y = torch.nn.functional.prelu(tensor_x, self.p)
        y = y.cpu().numpy()
        return {'y' : y}

    def postp(self, y_dict):
        # rand_t.shape = input.shape

        if self.rand_t is None:
            _shape = list(y_dict['y'].shape)
            for i in range(1, len(_shape), 1):
                _shape[i] = 1
            self.rand_t = np.random.rand(*_shape).astype('float32')
        z = y_dict['y'] * self.rand_t
        return {'z': z}


class Secure_Res(Secure_Layer):

    def __init__(self, layer_id, layer_name, input_layer, *args, **kwargs):
        super().__init__(layer_id, layer_name, input_layer, *args, **kwargs)
        self.type = 'res'


    def compute(self, _x):
        return {"y": np.sum(_x, 0, keepdims=0)}

    def postp(self, y_dict):

        if self.rand_t is None:
            bs = y_dict['y'].shape[0]
            self.rand_t = np.random.rand(bs).astype('float32')

        _shape = [1] * len(y_dict['y'].shape)
        _shape[0] = y_dict['y'].shape[0]
        z = y_dict["y"] * self.rand_t.reshape(*_shape)
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
