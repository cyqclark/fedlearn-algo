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

import argparse
import os,sys
import numpy as np
from typing import Callable, List, Dict

root_path = os.getcwd()

sys.path.append(root_path)
sys.path.append(os.path.join(root_path,'demos/HFL'))

from demos.HFL.example.oneflow.local_bert_classifier import train_classifier
from demos.HFL.example.oneflow.local_bert_classifier.train_classifier import oneFlowBertClassifier

from demos.HFL.base_client import Client
from demos.HFL.common.param_util import(
    Params,
    ParamsRes,
    TrainArgs,
    EvalArgs,
    TrainRes,
    EvalRes
)


class BertClient(Client):
    def __init__(self):
        load_pretrained_model =  True
        self.model = oneFlowBertClassifier(train_classifier.args, load_pretrained_model)


    def _init_params(self):
        W = self.__get_dummy_weights( num_layers =16)

        self.params = Params(
            names = list(W.keys()),
            weights= list(W.values()),
            weight_type= 'float'
        )    
        

    def get_params(self)->ParamsRes:
        
        param_dict: Dict[str,np.ndarray] = \
            self.model.get_model_parameters()
        
        return ParamsRes(
            Params(
                names=list(param_dict.keys()),
                weights=list(param_dict.values()),
                weight_type='float'
            ),
            response_messages={'dummy_res',1}
        )       
    
    
    def set_params(self, params:Params)->None:
        model_params = dict(zip(params.names, params.weights))
        self.model.load(model_params)
    
   
    def train(self, trainArgs:TrainArgs)->TrainRes:
        self.set_params(trainArgs.params)

        param_dict, acc = self.model.train()
        
        trainRes = TrainRes(
            params = Params(
                names=list(param_dict.keys()),
                weights=list(param_dict.values()),
                weight_type='float'
            ),
            num_samples= 100,
            metrics =  {'acc':acc}  
        )
        return trainRes
        

   
    def evaluate(self, evalAgrs:EvalArgs)->EvalArgs:
        pass
