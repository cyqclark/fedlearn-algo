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

__doc__ = "Local client model creation helper  "
import os,sys
import numpy as np 
import logging
import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(module) s - %(funcName) s - %(lineno) d - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

root_path = os.getcwd()

sys.path.append(os.path.join(root_path,'demos/HFL'))

from typing import Dict,Union

from core.entity.common.machineinfo import MachineInfo

from demos.HFL.base_client import Client


class ModelName():
    DUMMY_MODEL = 'dummy_model'
    ONEFLOW_BERT = 'oneflow_bert'
    PYTORCH_BERT = 'pytorch_bert'
    PYTORCH_BERT_TINY = 'pytorch_bert_tiny'
    PYTORCH_BERT_TextClassifier = 'pytorch_bert_text_classifier'
    PYTORCH_MNIST = 'pytorch_mnist'
    

def create_client(client_name:str, 
                 config:Dict[str, Union[float,int,str]]=None
                )->Client:
    """
    Create a local client given client's name 

    Parameters
    ----------
    client_name : str
        Local client's name defined in class ModelName.
    config : Dict
        Configuration for model

    Returns
    ----------
    client: Client 

    """
             
   
    if client_name == ModelName.ONEFLOW_BERT:
        from demos.HFL.example.oneflow.bert_client import BertClient
        return BertClient()
    
    elif client_name in [ModelName.PYTORCH_BERT,ModelName.PYTORCH_BERT_TINY,\
                ModelName.PYTORCH_BERT_TextClassifier]:
        from demos.HFL.example.pytorch.hugging_face.bert_client import BertClient
        from demos.HFL.example.pytorch.hugging_face.bert_client_text_classifier import BertClientTextClassifier
        huggingface_madel_map = {ModelName.PYTORCH_BERT:'bert-base-chinese',
                                 ModelName.PYTORCH_BERT_TINY:'uer/chinese_roberta_L-2_H-128',
                                 ModelName.PYTORCH_BERT_TextClassifier: 'bert-base-uncased'}
        if client_name == ModelName.PYTORCH_BERT_TextClassifier:
            return BertClientTextClassifier(
                config={
                    'pretrained_model_name': huggingface_madel_map[client_name]}
            )
        else:
            return BertClient(
                config={
                    'pretrained_model_name': huggingface_madel_map[client_name]}
            )
    elif client_name ==  ModelName.DUMMY_MODEL:
        from demos.HFL.example.dummy.dummy_client import DummyClient
        return DummyClient()        
    else: 
        raise ValueError(f"Model {client_name} not exists !")
    
            
