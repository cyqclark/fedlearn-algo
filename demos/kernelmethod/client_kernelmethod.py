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

from core.client.client import Client
from core.entity.common.message import RequestMessage, ResponseMessage
from core.entity.common.machineinfo import MachineInfo
from core.entity.common.modeltoken import generate_token
from kernelmethod import KernelMapping
from utils import Data

import logging
import numpy
import os
from typing import List


class ClientError(ValueError):
    pass


class KernelMethodClient(Client):
    def __init__(self, machine_info: MachineInfo):
        # pass arguments
        super().__init__(machine_info)
        self.algorithm_type = 'kernelmethod'
        self.source_data = None
        self.response = None
        self.machine_info = machine_info
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
        self.function_register()

    def function_register(self):
        # train
        self.dict_functions["train_init"] = self.train_init
        self.dict_functions["meta_comp"] = self.meta_comp
        self.dict_functions["train_loop_start"] = self.meta_comp
        self.dict_functions["param_update"] = self.param_update
        self.dict_functions["train_finish"] = self.training_finish
        # inference
        self.dict_functions["inference_init"] = self.inference_init
        self.dict_functions["inference_comp"] = self.prediction

    def normalization(self, norm_type: str):
        logging.info('Data normalization.')
        self.data.is_normalized = True
        self.data.normalize_fit(norm_type)
        self.data.normalize_transform()

    def load_inference_data(self, data_path: str):
        if os.path.exists(data_path) is False:
            raise ClientError('invalid data path.')
        self.data = Data(data_path)

    def load_data(self, data_path: str, feature_names:List, label_name:List = None):
        if os.path.exists(data_path) is False:
            raise ClientError('invalid data path.')

        self.data = Data(data_path)
        print(len(feature_names), 'dimension feature on machine ', self.machine_info.token)
        if feature_names is not None:
            self.data.feature = self.data.data_frame[feature_names].values

        if label_name is not None:
            self.data.label = self.data.data_frame[label_name].values
            self.is_active = True

    def train_init(self, request: RequestMessage) -> ResponseMessage:
        print('training initialization')
        sender_info = request.server_info
        receiver_info = request.client_info

        if self.kernel_mapping_param is None:
            raise ClientError('kernel mapping parameter is not properly initialized.')

        if self.data is None:
            raise ClientError('source data is not properly loaded.')

        self.kernel_mapping = KernelMapping(self.kernel_mapping_param)
        self.data.kernel_transform(self.kernel_mapping)

        self.map_dim = self.kernel_mapping.map_dim
        self.sample_num, self.feature_dim = self.data.feature.shape
        self.model_param = 0.1*numpy.ones((self.map_dim, 1), dtype=numpy.float)

        msg_body = {'sample_num': self.sample_num}
        response = ResponseMessage(sender=sender_info,
                                   receiver=receiver_info,
                                   body=msg_body,
                                   phase_id="res_tr_init_finish")
        return response

    def meta_comp(self, request: RequestMessage) -> ResponseMessage:
        print('compute meta result')
        sender_info = request.server_info
        receiver_info = request.client_info

        # clients compute inner product and send it to the server
        if self.is_active is True:
            meta_res = -1 * self.data.transformed_feature.dot(self.model_param)
            meta_res += self.data.label
        else:
            meta_res = -1 * self.data.transformed_feature.dot(self.model_param)
        msg_body = {'meta_result': meta_res}
        response = ResponseMessage(sender=sender_info,
                                   receiver=receiver_info,
                                   body=msg_body,
                                   phase_id="meta_comp")
        return response

    def param_update(self, request: RequestMessage) -> ResponseMessage:
        print('parameter update')
        sender_info = request.server_info
        receiver_info = request.client_info

        # server send aggregation to the clients then clients update local model parameter
        agg = request.body['aggregation_result']
        machine_id = request.body['chosen_machine']
        print(machine_id)
        print(self.machine_info.token == machine_id)
        if self.machine_info.token == machine_id:
            print('choose machine ', self.machine_info.token)
            agg += self.data.transformed_feature.dot(self.model_param)
            self.model_param = numpy.linalg.lstsq(self.data.transformed_feature, agg, rcond=None)[0]
            #print(agg)
            #print(self.model_param)
        msg_body = {}
        response = ResponseMessage(sender=sender_info,
                                   receiver=receiver_info,
                                   body=msg_body,
                                   phase_id="meta_comp")
        return response

    def training_finish(self, request: RequestMessage) -> ResponseMessage:
        sender_info = request.server_info
        receiver_info = request.client_info
        model_token = generate_token(self.algorithm_type, self.machine_info.token)

        self.model['scaler'] = self.data.scaler
        self.model['kernel_mapping_param'] = self.kernel_mapping_param
        self.model['model_param'] = self.model_param
        self.save_model(self.model_path+model_token + '.p', self.model)

        response = ResponseMessage(sender=sender_info,
                                   receiver=receiver_info,
                                   body={},
                                   phase_id="train_finish")
        return response

    def inference_init(self, request: RequestMessage) -> ResponseMessage:
        sender_info = request.server_info
        receiver_info = request.client_info
        print('inference initialization')

        if self.data is None:
            raise ClientError('source data is not properly set up.')
        self.kernel_mapping = KernelMapping(self.model['kernel_mapping_param'])

        self.data.normalize_transform(scaler=self.model['scaler'])
        self.data.kernel_transform(self.kernel_mapping)

        self.model_param = self.model['model_param']
        self.map_dim = self.kernel_mapping.map_dim
        self.sample_num, self.feature_dim = self.data.feature.shape
        msg_body = {'message': 'initialization_ready', 'sample_num': self.sample_num}
        response = ResponseMessage(sender=sender_info,
                                   receiver=receiver_info,
                                   body=msg_body,
                                   phase_id="inference_init")
        return response

    def prediction(self, request: RequestMessage) -> ResponseMessage:
        sender_info = request.server_info
        receiver_info = request.client_info

        inner_product = self.data.transformed_feature.dot(self.model_param)
        msg_body = {'inner_product': inner_product}
        response = ResponseMessage(sender=sender_info,
                                   receiver=receiver_info,
                                   body=msg_body,
                                   phase_id="inference_comp")

        return response
