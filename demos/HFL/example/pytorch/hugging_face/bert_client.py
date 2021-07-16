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

import os,sys
import numpy as np
import time
from typing import Callable, List, Dict, Union

root_path = os.getcwd()

sys.path.append(root_path)
sys.path.append(os.path.join(root_path,'core/entity'))
sys.path.append(os.path.join(root_path,'demos/HFL'))

BERT_ROOT = os.path.join(root_path,'demos/HFL/example/pytorch/hugging_face')

from demos.HFL.example.pytorch.hugging_face.local_bert_lm.model import (
    BertMlM_Model,
    DataConfig,
    TrainConfig
)

from demos.HFL.base_client import Client
from demos.HFL.common.param_util import(
    Params,
    ParamsRes,
    TrainArgs,
    EvalArgs,
    TrainRes,
    EvalRes,
    NLPInferArgs,
    NLPInferRes
)


_TRAIN_DATA_FILE_='web_text_zh_testa.json'

def get_train_args():
        train_data_file = f'{BERT_ROOT}/{_TRAIN_DATA_FILE_}'
        #extension = train_data_file.split(".")[-1]
        data_config = DataConfig(
            train_file=train_data_file,
            val_file=train_data_file,
            mlm_probability=0.15,
            max_train_samples=10000,
            max_eval_samples=1000,
            pad_to_max_length=True,
            line_by_line=True
        )
        train_config = TrainConfig(
            output_dir=f'{BERT_ROOT}/saved_tiny_models',
            do_train=True,
            do_eval=True,
            report_to='tensorboard',
            logging_dir=f'{BERT_ROOT}/log',
            evaluation_strategy="steps",
            num_train_epochs=1,
            eval_steps=1000,
            per_device_train_batch_size=6)
            
        return data_config, train_config,     

class BertClient(Client):
    
    def __init__(self,
                config:Dict[str,Union[str,float,int]]=None):
        
        pretrained_model_name = \
            config.get('pretrained_model_name','bert-base-chinese') if config is not None else 'bert-base-chinese'
        
        self.model = BertMlM_Model(
            *get_train_args(),
            pretrained_model_name=pretrained_model_name) 

        
    def get_params(self)->ParamsRes:
        
        param_dict: Dict[str,np.ndarray] = \
            self.model.get_model_parameters()
        
        return ParamsRes(
            Params(
                list(param_dict.keys()),
                list(param_dict.values()),
                weight_type='float'),
            response_messages={'dummy_msg',1})       
    
    
    def set_params(self, params:Params)->None:
        model_params = dict(zip(params.names,params.weights))
        self.model.set_model_parameters(model_params)
    
   
    def train(self, trainArgs:TrainArgs)->TrainRes:
        self.set_params(trainArgs.params)
        param_dict, metrics  = self.model.train()
        
        trainRes = TrainRes(
            params = Params(
                names=list(param_dict.keys()),
                weights=list(param_dict.values()),
                weight_type='float'),
            num_samples= metrics['train_samples'],
            metrics =  metrics  
        )
        return trainRes


    def inference(self, inputArgs: NLPInferArgs) -> NLPInferRes:
        if inputArgs.params is not None:
           self.set_params(inputArgs.params)

        predicts = self.model.inference(inputArgs.inputs)
        inferRes = NLPInferRes(
            inputArgs.inputs,
            outputs=predicts
        )
        return inferRes   

   
    def evaluate(self, evalAgrs:EvalArgs)->EvalArgs:
        pass
