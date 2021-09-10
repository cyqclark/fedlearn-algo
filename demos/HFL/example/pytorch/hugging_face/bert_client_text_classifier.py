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
BERT_ROOT = os.path.join(root_path,'demos/HFL/example/pytorch/hugging_face')

from demos.HFL.example.pytorch.hugging_face.bert_client import BertClient
from demos.HFL.example.pytorch.hugging_face.local_bert_text_classifier.model import (
    BertTextClassifier,
    DataConfig,
    TrainConfig
)

_TRAIN_DATA_FILE_='data/demo_train_data.txt'

def get_train_args():
        train_data_file = f'{BERT_ROOT}/{_TRAIN_DATA_FILE_}'
        #extension = train_data_file.split(".")[-1]
        data_config = DataConfig(
            train_file=train_data_file,
            val_file=train_data_file,
            mlm_probability=0.15,
            max_train_samples=50000, #4500,
            max_eval_samples=5000, #1000,
            pad_to_max_length=True,
            line_by_line=True
        )
        train_config = TrainConfig(
            output_dir=f'{BERT_ROOT}/saved_text_classifier_models',
            do_train=True,
            do_eval=True,
            report_to='tensorboard',
            logging_dir=f'{BERT_ROOT}/log',
            num_train_epochs=1,
            #num_train_epochs=10,
            #per_device_train_batch_size=6
            per_device_train_batch_size=10,
            per_device_eval_batch_size=20,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_steps=100,               # log & save weights each logging_steps
            logging_strategy="steps",
            #evaluation_strategy="steps",      # evaluate each `logging_steps`
            eval_steps=100,
            evaluation_strategy="steps",      # evaluate each `logging_steps`
            #evaluation_strategy="epoch",      # evaluate each `logging_steps`
        )
            
        return data_config, train_config,     

class BertClientTextClassifier(BertClient):
    
    def __init__(self,
                config:Dict[str,Union[str,float,int]]=None):
        
        pretrained_model_name = \
            config.get('pretrained_model_name','bert-base-chinese') if config is not None else 'bert-base-chinese'
        
        self.model = BertTextClassifier(
            *get_train_args(),
            pretrained_model_name=pretrained_model_name) 
        print('\t', self.model)        

        
