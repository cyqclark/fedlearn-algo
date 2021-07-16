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

import os
import math
import numpy as np

import oneflow as flow
import pickle

from omegaconf import open_dict
from classifier_util import GlueBERT
from util import Snapshot, InitNodes, Metric, CreateOptimizer, GetFunctionConfig

import config as configs
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score
from typing import List, Dict


from config import logger


args = configs.get_config()

dataset = "CoLA"

if dataset == "CoLA":
#   train_example_num=8551
#   eval_example_num=1043
  train_example_num=370000//4
  eval_example_num=10430
  test_example_num=1063
  learning_rate=1e-5
  wd=0.01
else:
  train_example_num=3668
  eval_example_num=408
  test_example_num=1725
  learning_rate=2e-6
  wd=0.001

with open_dict(args):
    args.task_name = 'CoLA'
    args.num_epochs = 3
    args.train_data_prefix = 'train.of_record-'
    args.train_example_num = 370000//4
    args.train_data_part_num  = 1
    args.eval_data_prefix = 'eval.of_record-'
    args.eval_example_num = 10833
    args.eval_batch_size_per_device = 64
    args.eval_data_part_num = 1
    args.label_num = 15
    #args.label_num = 2
    
    # ----------- Model and Data Path ----------------------------------
    #Root_Dir = '/data/tzeng/source/project/OneFlow-Benchmark/LanguageModeling/BERT'
    Root_Dir = '../OneFlow-Benchmark/LanguageModeling/BERT'
    # dataset = 'dataset/glue_ofrecord/CoLA'
    dataset ='toutiao-text-classfication-dataset/of_data'
    args.train_data_dir=f'{Root_Dir}/{dataset}/train'
    args.eval_data_dir=f'{Root_Dir}/{dataset}/eval'
    # args.model_load_dir=f'{Root_Dir}/bert_model/uncased_L-12_H-768_A-12/uncased_L-12_H-768_A-12-oneflow'
    args.model_load_dir=f'{Root_Dir}/bert_model/chinese_L-12_H-768_A-12/chinese_L-12_H-768_A-12-oneflow'
    args.model_save_dir=f'./snapshots'
    args.gpu_num_per_node=3
    args.num_epochs=1
    args.eval_example_num=eval_example_num
    # args.eval_batch_size_per_device = 4
    args.loss_print_every_n_iter = 20
    args.batch_size_per_device=32
    args.loss_print_every_n_iter=20
    args.save_last_snapshot=True
    args.seq_length=64

    #args.seq_length=128
    args.num_hidden_layers=12
    args.num_attention_heads=12
    args.max_position_embeddings=512
    args.type_vocab_size=2
    args.vocab_size=21128
    args.attention_probs_dropout_prob=0.1
    args.hidden_dropout_prob=0.1 
    args.hidden_size_per_head=64
    args.learning_rate=learning_rate
    args.weigt_decay=wd

    args.batch_size = args.num_nodes * args.gpu_num_per_node * args.batch_size_per_device
    args.eval_batch_size = args.num_nodes * args.gpu_num_per_node * args.eval_batch_size_per_device
    args.epoch_size = math.ceil(args.train_example_num / args.batch_size)
    args.num_eval_steps = math.ceil(args.eval_example_num / args.eval_batch_size)
    args.iter_num = args.epoch_size * args.num_epochs

    # ---- Fedlearn Deep Model weights obtain-compute-update params ----#
    TEST_WEIGHT_GET_UPDATE = True


configs.print_args(args)


def BertDataDecoder(
    data_dir, batch_size, data_part_num, seq_length, part_name_prefix, shuffle=True
):
    with flow.scope.placement("cpu", "0:0"):
        ofrecord = flow.data.ofrecord_reader(data_dir,
                                             batch_size=batch_size,
                                             data_part_num=data_part_num,
                                             part_name_prefix=part_name_prefix,
                                             random_shuffle=shuffle,
                                             shuffle_after_epoch=shuffle)
        blob_confs = {}
        def _blob_conf(name, shape, dtype=flow.int32):
            blob_confs[name] = flow.data.OFRecordRawDecoder(ofrecord, name, shape=shape, dtype=dtype)

        _blob_conf("input_ids", [seq_length])
        _blob_conf("input_mask", [seq_length])
        _blob_conf("segment_ids", [seq_length])
        _blob_conf("label_ids", [1])
        #_blob_conf("is_real_example", [1])

        return blob_confs


def BuildBert(
    batch_size,
    data_part_num,
    data_dir,
    part_name_prefix,
    args,
    shuffle=True
):
    hidden_size = 64 * args.num_attention_heads  # , H = 64, size per head
    intermediate_size = hidden_size * 4

    decoders = BertDataDecoder(
        data_dir, batch_size, data_part_num, args.seq_length, part_name_prefix, shuffle=shuffle
    )
    #is_real_example = decoders['is_real_example']

    loss, logits = GlueBERT(
        decoders['input_ids'],
        decoders['input_mask'],
        decoders['segment_ids'],
        decoders['label_ids'],
        args.vocab_size,
        seq_length=args.seq_length,
        hidden_size=hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_act="gelu",
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
        max_position_embeddings=args.max_position_embeddings,
        type_vocab_size=args.type_vocab_size,
        initializer_range=0.02,
        label_num=args.label_num
    )
    return loss, logits, decoders['label_ids']


@flow.global_function(type='train', function_config=GetFunctionConfig(args))
def BertGlueFinetuneJob():
    loss, logits, _ = BuildBert(
        args.batch_size,
        args.train_data_part_num,
        args.train_data_dir,
        args.train_data_prefix,
        args
    )
    
    flow.losses.add_loss(loss)
    opt = CreateOptimizer(args)
    opt.minimize(loss)
    #frozen_bottom_layers()
    return {'loss': loss}


@flow.global_function(type='predict', function_config=GetFunctionConfig(args))
def BertGlueEvalTrainJob():
    _, logits, label_ids = BuildBert(
        args.batch_size,
        args.train_data_part_num,
        args.train_data_dir,
        args.train_data_prefix,
        args,
        shuffle=False
    )
    return logits, label_ids


@flow.global_function(type='predict', function_config=GetFunctionConfig(args))
def BertGlueEvalValJob():
    #8551 or 1042
    _, logits, label_ids = BuildBert(
        args.eval_batch_size,
        args.eval_data_part_num,
        args.eval_data_dir,
        args.eval_data_prefix,
        args,
        shuffle=False
    )
    return logits, label_ids



def run_eval_job(eval_job_func, num_steps, desc='train'):
    labels = []
    predictions = []
    for index in range(num_steps):
        logits, label = eval_job_func().get()
        predictions.extend(list(logits.numpy().argmax(axis=1)))
        labels.extend(list(label))

    def metric_fn(predictions, labels):
        return {
            "accuracy": accuracy_score(labels, predictions), 
            "matthews_corrcoef": matthews_corrcoef(labels, predictions), 
            "precision": precision_score(labels, predictions, average='macro'), 
            "recall": recall_score(labels, predictions,average='macro'),
            "f1": f1_score(labels, predictions,average='macro'),
        }

    metric_dict = metric_fn(predictions, labels)
    print(desc, ', '.join('{}: {:.3f}'.format(k, v) for k, v in metric_dict.items()))
    return metric_dict


def get_bert_variable(adm_param_filter=True):
    variables = flow.get_all_variables()
    
    
    def filter_cond(name):
        return adm_param_filter and name.strip()[-2:] in['-v','-m'] or name == 'System-Train-TrainStep-BertGlueFinetuneJob'      
    
    logger.debug("Abtain Bert model parameters ... ")
    V = {name: data.numpy() for name, data in variables.items() if not filter_cond(name)}
    return V


class oneFlowBertClassifier():
    def __init__(self, config, load_model=True):
        self.CFGs = config
        args = self.CFGs
        flow.config.gpu_device_num(args.gpu_num_per_node)
        flow.env.log_dir(args.log_dir)
        InitNodes(args)
        logger.debug("oneFlowBertClassifier -- 完成初始化...")
        
        if load_model:
            self.snapshot = Snapshot(self.CFGs.model_save_dir, self.CFGs.model_load_dir)
        else: 
            # save model ...
            self.snapshot = Snapshot(self.CFGs.model_save_dir, None)    
    
    def eval(self):
        eval_metrics = run_eval_job(BertGlueEvalValJob, self.CFGs.num_eval_steps, desc='eval')
        return eval_metrics['accuracy']
    
    def get_model_parameters(self)->Dict[str, np.ndarray]:
        return  get_bert_variable()

    def load(self, modelPara):
        flow.load_variables(modelPara)

    def train(self):
        #print('Start Train oneFlowBert Classifier ...')
        logger.debug("Start Train oneFlowBert Classifier ...")
        
        for epoch in range(self.CFGs.num_epochs):    
            metric = Metric(desc='finetune', print_steps=self.CFGs.loss_print_every_n_iter,  
                        batch_size=self.CFGs.batch_size, keys=['loss'])
        
            for step in range(self.CFGs.epoch_size):
                BertGlueFinetuneJob().async_get(metric.metric_cb(step, epoch=epoch))
                #if 1: #step % args.loss_print_every_n_iter == 0: 
            run_eval_job(BertGlueEvalTrainJob, self.CFGs.epoch_size, desc='train')
            eval_metrics = run_eval_job(BertGlueEvalValJob, self.CFGs.num_eval_steps, desc='eval')
           
        
        if self.CFGs.save_last_snapshot:
            try:
                self.snapshot.save("last_snapshot")
            except Exception as e:
                logger.debug(e)
        
        dummy_train_samples = 999

        if TEST_WEIGHT_GET_UPDATE:
            model_varaibls = get_bert_variable()
            eval_metrics['acc']= eval_metrics['accuracy']
            eval_metrics['train_samples']= dummy_train_samples
            return   model_varaibls,  eval_metrics 




def test():
    with open_dict(args):
        args.batch_size = args.num_nodes * args.gpu_num_per_node * args.batch_size_per_device
        args.eval_batch_size = args.num_nodes * args.gpu_num_per_node * args.eval_batch_size_per_device

        args.epoch_size = math.ceil(args.train_example_num / args.batch_size)
        args.num_eval_steps = math.ceil(args.eval_example_num / args.eval_batch_size)
        args.iter_num = args.epoch_size * args.num_epochs

        # ----------- Model and Data Path ----------------------------------
        # Root_Dir = '/data/tzeng/source/project/OneFlow-Benchmark/LanguageModeling/BERT'
        Root_Dir = '../LanguageModeling/BERT'
        dataset = 'dataset/glue_ofrecord/CoLA'

        args.train_data_dir=f'{Root_Dir}/{dataset}/train'
        args.eval_data_dir=f'{Root_Dir}/{dataset}/eval'
        args.model_load_dir=f'{Root_Dir}/bert_model/uncased_L-12_H-768_A-12/uncased_L-12_H-768_A-12-oneflow'
        args.model_save_dir=f'./snapshots'

    configs.print_args(args)

    trainer = oneFlowBertClassifier(args)
    
    trainer.train()


if __name__ == '__main__':
    test()