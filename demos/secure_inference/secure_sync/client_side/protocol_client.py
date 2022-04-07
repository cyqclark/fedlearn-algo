import numpy as np
# from utils.data_transfer import serialize, deserialize
import requests
import time

class Secure_Layer(object):

    def __init__(self, *args, **kwargs):
        self.config = kwargs
        self.secure_inference_url = args[0]

    def prep(self, x):
        pass

    def server_call(self, _x):
        pass

    def postp(self, msg_dict):
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

    def prep(self, x):
        # x.shape = (bs, ch, w, h) or (bs, ch)
        if len(x.shape) == 4:
            # for image format:
            # x.shape = bs, ch, w, h
            bs, ch, w, h = x.shape
            p = np.random.randn(bs, self.shard, 1, 1, 1)#TODO: double check here
            q = 100 * np.random.randn(bs, self.shard, ch, w, h)
            q[:, -1:] = - np.sum(q[:, :-1], 1, keepdims=1)
            _x = np.tile( np.expand_dims(x, 1), (1,self.shard,1,1,1) )
            perturbed_x = p * ( _x + q )
            self.I = np.reciprocal(p) / self.shard#np.sum(p, 1, keepdims=1)

        elif len(x.shape) == 2:
            bs, ch = x.shape
            p = np.random.randn(bs, self.shard, 1)
            q = 10 * np.random.randn(bs, self.shard, ch)
            q[:, -1:] = - np.sum(q[:, :-1], 1, keepdims=1)

            _x = np.tile( np.expand_dims(x, 1), (1,self.shard,1) )
            perturbed_x = np.tile(p, (1,1,ch)) * ( _x + q )
            self.I = np.reciprocal(p) / self.shard #np.sum(p, 1, keepdims=1)

        else:
            raise(f"x shape {x.shape} not supported!")

        return perturbed_x

    def postp(self, res_dict):
        # get input activation
        _z, _b = np.asarray(res_dict['_z']), np.asarray(res_dict['_b'])
        # assemble result 
        if len(_z.shape) == 5:
            bs, shard, ch_out, w_out, h_out = _z.shape
            tmp = _z * np.tile(self.I, (1,1,ch_out,w_out,h_out)) # element-wise division
            ax_dot_t = np.sum(tmp, axis=1, keepdims=0) # (bs, ch_out, w_out, h_out)
            b_dot_t = _b.reshape(bs, ch_out, 1, 1) # (bs, ch_out, w_out, h_out)
            y_dot_t = ax_dot_t + b_dot_t # (bs, ch_out, w_out, h_out)

        elif len(_z.shape) == 3:
            bs, shard, ch_out = _z.shape
            tmp = _z * np.tile(self.I, (1,1,ch_out)) # element-wise division
            ax_dot_t = np.sum(tmp, axis=1, keepdims=0) # (bs, ch_out)
            b_dot_t = _b.reshape(bs, ch_out) # (bs, ch_out)
            y_dot_t = ax_dot_t + b_dot_t # (bs, ch_out)

        else:
            print(f"server return format error! return shape {z.shape}")

        return y_dot_t


class Secure_Nonlinear(Secure_Layer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prep(self, x):
        # perturb activation
        p = np.random.randn(*x.shape)
        p = np.zeros_like(p)
        self.pos_p = np.abs(p) + 1
        x_prim = x * self.pos_p
        return x_prim

    def postp(self, res_dict):
        z = np.asarray(res_dict['z'])
        # save result
        return z / self.pos_p
  
 
class Secure_Res(Secure_Layer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shard = self.config['shard']

    def prep(self, x_list):
        _shape = x_list[0].shape
        self.p = np.random.randn(*_shape)
        perturbed_x = np.asarray([self.p * xi for xi in x_list])
        return perturbed_x 

    def postp(self, res_dict):
        z = np.asarray(res_dict['z'])
        return  z / self.p

 
class Secure_Reshape(Secure_Layer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shard = self.config['shard']

    def prep(self, x):
        _shape = x.shape
        self.p = np.random.randn(*_shape)
        perturbed_x = self.p * x
        return perturbed_x 

    def postp(self, res_dict):
        z = np.asarray(res_dict['y'])
        return  z / self.p.reshape(*z.shape)


class SP_Client(object):

    def __init__(self, compute_graph, shard):
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

        self.activation = {} #

    def get_layer(self, _id):

        def wraper(cur_layer_id=_id):
            input_layer_id = self.compute_graph[cur_layer_id]['input']
            if isinstance(input_layer_id, list):
                x = [self.activation[i] for i in input_layer_id]
            else:
                x = self.activation[input_layer_id]

            cur_layer = self.sp_graph[cur_layer_id]
            self.activation[cur_layer_id] = cur_layer.forward(x)  # TODO: replace with RPC call
            return self.activation[cur_layer_id]

        return wraper


    def prep(self, cur_layer_id):
        input_layer_id = self.compute_graph[cur_layer_id]['input']
        if isinstance(input_layer_id, list):
            x = [self.activation[i] for i in input_layer_id]
        else:
            x = self.activation[input_layer_id]

        cur_layer = self.sp_graph[cur_layer_id]
        return cur_layer.prep(x)

    def postp(self, x, cur_layer_id):
        cur_layer = self.sp_graph[cur_layer_id]
        self.activation[cur_layer_id] = cur_layer.postp(x)
        return self.activation[cur_layer_id]


