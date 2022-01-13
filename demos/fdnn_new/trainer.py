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

__doc__ = "Train helper module for vertical deep learning"
import torch
from torch import nn
from typing import Dict
import numpy
from utils import HyperParameters
from table_data import TableData, MiniBatchData


class ActiveTrainer:
    def __init__(self, model, hyper_parameters: HyperParameters):
        self.hyper_parameters = hyper_parameters
        self.loss = nn.BCELoss()
        self.model = model
        self.sample_number = None
        self.batch_index = 0
        self.batch_size = None

    def generate_mini_batch(self, data: TableData, batch_index: int) -> MiniBatchData:
        # batch_index starts from 1
        start = (batch_index - 1) * self.batch_size
        end = max(batch_index * self.batch_size, self.sample_number)
        return data.get_minibatch([i for i in range(start, end)])

    def model_forward(self, data: MiniBatchData, remote_input):
        return self.model(data.cate_data, data.num_data, remote_input)

    def model_backward(self, out, y):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyper_parameters.lr, weight_decay=1e-4)
        optimizer.zero_grad()
        loss_val = self.loss(out, torch.from_numpy(numpy.expand_dims(y, axis=1).astype("float32")))
        loss_val.backward()
        print("loss function value is ", loss_val)
        optimizer.step()
        return self.model
   
    def get_remote_gradients(self):
        grad = {}
        for client in self.model.remote_clients:
            grad[client] = self.model.remote_input[client].grad
        return grad


class PassiveTrainer:
    def __init__(self, model, hyper_parameters: HyperParameters):
        self.model = model
        self.hyper_parameters = hyper_parameters
        self.output = None
        self.sample_number = None
        self.batch_index = 0
        self.batch_size = None 
        
    def generate_mini_batch(self, data: TableData, batch_index: int) -> MiniBatchData:
        # batch_index starts from 1
        start = (batch_index - 1) * self.batch_size
        end = max(batch_index * self.batch_size, self.sample_number)
        return data.get_minibatch([i for i in range(start, end)])

    def model_forward(self, data: MiniBatchData):
        self.output = self.model(data)
        return

    def model_backward(self, grad):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyper_parameters.lr, weight_decay=1e-4)
        optimizer.zero_grad()
        self.output.backward(grad)
        optimizer.step()
       
    def get_output(self):
        """
        Obtain the output of the model.
        """
        return self.output.clone().detach().requires_grad_()
    

