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
from collections import OrderedDict
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union,Dict,List,Any

from datasets import load_dataset

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
import math

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version


from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    BertTokenizerFast,
    AlbertModel
    )
import os    
import json
import argparse

#_TRAIN_DATA_FILE_='web_text_zh_testa.json'

@dataclass
class DataConfig():
    train_file:str
    val_file:str
    mlm_probability:float
    max_train_samples:int
    max_eval_samples:int
    pad_to_max_length:bool =True
    line_by_line:bool = True
    preprocessing_num_workers:int = 8
    max_seq_length:int = 256
    

TrainConfig = TrainingArguments

_ROOTDIR_ = 'demos/HFL/example/pytorch/hugging_face'
class BertMlM_Model():
    TEXT_FIELD = 'text'
    # MODEL_NAME_PAIR = {'bert-base-chinese':BertForMaskedLM, 'clue/albert_chinese_tiny':} 
    MODEL_NAME = ['bert-base-chinese','uer/chinese_roberta_L-2_H-128']
    def __init__(self,
                 data_config:DataConfig,
                 train_config:TrainConfig,
                 pretrained_model_name='bert-base-chinese',
                 init_from_local_pretrained=False):

        self.data_config = data_config
        self.train_config =  train_config

        self.train_config.output_dir = os.path.join(
              self.train_config.output_dir,
              pretrained_model_name
        )
        
        # self.train_config.output_dir = \
            # '/'.join(self.train_config.output_dir.split('/')[:-1]) \
                # + '/'+ pretrained_model_name
        

        if pretrained_model_name not in BertMlM_Model.MODEL_NAME:
            raise(ValueError(f"{pretrained_model_name} is not a supported pretrained model"))
        

        self.tokenizer =  \
            BertTokenizerFast.from_pretrained(pretrained_model_name)
        
        model_name = pretrained_model_name \
            if not init_from_local_pretrained \
            else  f'{_ROOTDIR_}/{pretrained_model_name}'
        
        self.model = \
            BertForMaskedLM.from_pretrained(model_name)
        
        if self.train_config.do_train or self.train_config.do_eval:
            self.tokenized_datasets = self._load_dataset()
            

            pad_to_multiple_of_8 = self.data_config.line_by_line and  self.train_config.fp16 and not self.data_config.pad_to_max_length
            self.data_collator = DataCollatorForLanguageModeling(
                                    tokenizer=self.tokenizer,
                                    mlm_probability=self.data_config.mlm_probability,
                                    mlm = True,
                                    pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
                                )

        if self.train_config.do_train:
            if "train" not in self.tokenized_datasets:
                raise ValueError("--do_train requires a train dataset")
            self.train_dataset = self.tokenized_datasets["train"].shuffle()
            if data_config.max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(range(self.data_config.max_train_samples))

        if self.train_config.do_eval:
            if "validation" not in self.tokenized_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            self.eval_dataset = self.tokenized_datasets["validation"]
            if self.data_config.max_eval_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(self.data_config.max_eval_samples))        
                           
    
    def train(self):
        trainer = Trainer(
            model=self.model,
            args=self.train_config,
            train_dataset=self.train_dataset if self.train_config.do_train else None,
            eval_dataset=self.eval_dataset if self.train_config.do_eval else None,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
        
        num_train_sample = 1 
        
        # Training
        #if self.train_config.do_train:
        
        checkpoint = self.train_config.resume_from_checkpoint    
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        
        metrics = train_result.metrics

        max_train_samples = self.data_config.max_train_samples \
                if self.data_config.max_train_samples is not None \
                else len(self.train_dataset)
        metrics["train_samples"] = min(max_train_samples, len(self.train_dataset))
        
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
            
            
        
        num_train_sample = metrics["train_samples"]
        #if self.train_config.do_eval:
        # logger.info("*** Evaluate ***")
        
        # Evaluation
        metrics = trainer.evaluate()

        max_eval_samples = self.data_config.max_eval_samples if self.data_config.max_eval_samples is not None else len(self.eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(self.eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        
        metrics["perplexity"] = perplexity
        

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        
        return_matrics ={
            'eval_loss':metrics["eval_loss"],
            'perplexity':metrics['perplexity'],
            'train_samples':num_train_sample 
        }

        
        
        return self.get_model_parameters(), return_matrics

    def inference(self, inputs_sents:List[str])->Dict[str,List[Any]]:
        """ Retreive CLS vector (last layer's first vector), shape: [num_sents, vec_length:768 for regular Bert] """   
        self.model.eval()
       
        outputs = self.model(**self.tokenizer(
                                            inputs_sents,
                                            return_tensors="pt",
                                            padding=True),
                            output_hidden_states=True)['hidden_states'][-1][:,0]
        np_list=[d.detach().cpu().numpy() for d in outputs]
        return {'CLS': np_list}                    

    def _load_dataset(self):
        extension = self.data_config.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        
        datasets = \
            load_dataset(
                extension, 
                data_files={'train':self.data_config.train_file,
                            'validation':self.data_config.val_file},
                cache_dir='./cache'                        

                )
        
        padding = "max_length" if self.data_config.pad_to_max_length else False

        def tokenize_function(examples):
            field = BertMlM_Model.TEXT_FIELD
            examples[field] = [str(line) for line in examples[field] if len(line) > 0 and not line.isspace()]
            return self.tokenizer(
                examples[field],
                padding=padding,
                truncation=True,
                max_length=self.data_config.max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            #num_proc=self.data_config.preprocessing_num_workers,
            num_proc=1,
            load_from_cache_file=False
            #remove_columns=['qid', 'title', 'desc', 'topic', 'star', 'answer_id', 'answerer_tags']
            #remove_columns=['news_id', 'keywords', 'desc', 'title', 'source', 'time']
        )        
        return tokenized_datasets           
    
    
    def set_model_parameters(self, 
                            params:Dict[str,np.ndarray],
                            mode='train'):
        if mode =='train':
            self.model.train()
        else:
            self.model.eval()
                 
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params.items()})
        self.model.load_state_dict(state_dict, strict=False)

    
    def get_model_parameters(self)->Dict[str,np.ndarray]:
        return {
            name: val.cpu().numpy()
            for name, val in self.model.state_dict().items()
            #if "bn" not in name
            }
    
    def eval(self):
        pass



def get_configs():
    
    config_file ='local_config.json'        
    with open(os.path.join(_ROOTDIR_,os.path.join('local_bert_lm',config_file))) as f:
        configs = json.load(f)

    # configs['train_config']['output_dir'] = os.path.join(_ROOTDIR_, configs['train_config']['output_dir']) 
    # 
    input_configs = {'data_config': DataConfig(**configs['data_config']),
                'train_config':TrainConfig(**configs['train_config'])} 
    for k,v in configs.items():
        if k not in ['data_config','train_config']:
            input_configs[k] = v
    
    return input_configs                            
    
 
    
def local_train():
    model = BertMlM_Model(**get_configs())
    model.train()


    text_test_file = f'{_ROOTDIR_}/test_sentences.txt'
    text_output_file = f'{_ROOTDIR_}/output_test_sentences.txt'
    texts = [line.strip() for line in  open(text_test_file,'r')]
    with torch.no_grad():
        cls_represnts:Dict[str,List] = model.inference(inputs_sents=texts)

    with open(text_output_file ,'w') as f:
        for v in cls_represnts['CLS']:
            f.write(str(v.tolist()) +'\n')
def local_predict():
    
    configs = get_configs()
    configs['train_config'].do_eval =False 
    configs['train_config'].do_train=False
    
    model = BertMlM_Model(**configs,
                        #pretrained_model_name='bert-base-chinese',
                        init_from_local_pretrained=True)

    text_test_file = f'{_ROOTDIR_}/test_sentences.txt'
    text_output_file = f'{_ROOTDIR_}/output_test_sentences.txt'
    texts = [line.strip() for line in  open(text_test_file,'r')]
    with torch.no_grad():
        cls_represnts:Dict[str,List] = model.inference(inputs_sents=texts)

    with open(text_output_file ,'w') as f:
        for v in cls_represnts['CLS']:
            f.write(str(v.tolist()) +'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_or_train', type=str,default='train',choices=['predict','train'])
    
    args = parser.parse_args()
    if args.predict_or_train == 'predict':
        local_predict()
    else: 
        local_train()    
