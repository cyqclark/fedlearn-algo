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
from dataclasses import dataclass, field
import os
import datetime
from typing import Optional, Union,Dict,List,Any
from collections import OrderedDict
import logging
import torch
import transformers
#from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers.file_utils import is_torch_available
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np
import random

from demos.HFL.example.pytorch.hugging_face.local_bert_text_classifier.dataset import read_20newsgroups

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logger = logging.getLogger(__name__)
transformers.utils.logging.set_verbosity_info()

model_path = "20newsgroups-bert-base-uncased"
# Log on each process the small summary:

data_file = 'iid_20newsgroups_0.csv'
data_file = 'iid_20newsgroups_1.csv'
data_file = 'noniid_quantity_20newsgroups_beta0.1_0.csv'
#data_file = 'noniid_quantity_20newsgroups_beta0.9_1.csv'
data_file = 'noniid_label_20newsgroups_alpha0.5_0.csv'
#data_file = 'noniid_label_20newsgroups_alpha0.5_1.csv'
test_file = 'test_20newsgroups.csv'

#data_file = None
if 1:
    def set_seed(seed: int):
        """
        Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
        installed).

        Args:
            seed (:obj:`int`): The seed to set.
        """
        random.seed(seed)
        np.random.seed(seed)
        if is_torch_available():
            print('using pytorch...')
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    set_seed(1)

    (train_texts, valid_texts, train_labels, valid_labels), target_names = read_20newsgroups(data_file=data_file, test_file=test_file)
    print('\t==========dataset===========')
    print('\ttrain samples: ', len(train_texts), type(train_texts))
    print('\tvalid samples: ',len(valid_texts), type(valid_texts))
    #print('>>', train_texts[0])

    # the model we gonna train, base uncased BERT
    # check text classification models here: https://huggingface.co/models?filter=text-classification
    model_name = "bert-base-uncased"
    # load the tokenizer
    tokenizer = transformers.BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)

    # max sequence length for each document/sentence sample
    max_length = 512
    # tokenize the dataset, truncate when passed `max_length`,
    # and pad with 0's when less than `max_length`
    #train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length, is_split_into_words=True)
    #valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length, is_split_into_words=True)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)

    class NewsGroupsDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor([self.labels[idx]])
            return item

        def __len__(self):
            return len(self.labels)

    # convert our tokenized data into a torch Dataset
    train_dataset = NewsGroupsDataset(train_encodings, train_labels)
    eval_dataset = NewsGroupsDataset(valid_encodings, valid_labels)

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

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
def compute_metrics(pred):
      true_labels = pred.label_ids
      true_predictions = pred.predictions.argmax(-1)
      # calculate accuracy using sklearn's function
      return {
            "accuracy_score": accuracy_score(true_labels, true_predictions),
            #"precision": precision_score(true_labels, true_predictions),
            #"recall": recall_score(true_labels, true_predictions),
            #"f1": f1_score(true_labels, true_predictions),
      }

class BertTextClassifier():
    def __init__(self,
                 data_config:DataConfig,
                 train_config:TrainingArguments,
                 pretrained_model_name='bert-base-chinese',
                 init_from_local_pretrained=False):

        self.data_config = data_config
        self.train_config =  train_config
        now = datetime.datetime.now()
        #self.train_day = now.strftime("%m/%d/%Y%H%M%S")
        self.train_day = now.strftime("%Y%m%d")

        self.train_config.output_dir = os.path.join(
              self.train_config.output_dir,
              pretrained_model_name
        )

        # load the model and pass to CUDA
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(target_names)).to(device)
        # load the tokenizer
        self.tokenizer = transformers.BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)

    def train(self):
        print('\t========BertTextClassifier=========', )

        trainer = Trainer(
            model=self.model,                    # the instantiated Transformers model to be trained
            args=self.train_config,              # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=eval_dataset,          # evaluation dataset
            compute_metrics=compute_metrics,     # the callback that computes metrics of interest
        )

        # train the model
        print('\t>>>train the local model')
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics
        logger.info(train_result)
        metrics['training_loss'] = train_result.training_loss
        metrics['global_step'] = train_result.global_step
        #logger.info(metrics)

        max_train_samples = self.data_config.max_train_samples \
                if self.data_config.max_train_samples is not None \
                else len(train_dataset)
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        #num_train_sample = metrics["train_samples"]

        # evaluate the current model after training
        print('\t>>>evaluate the local model')
        results = trainer.evaluate()
        max_eval_samples = self.data_config.max_eval_samples if self.data_config.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        output_eval_file = os.path.join(self.train_config.output_dir, self.train_day+"_localmodel_results_text_classifier.txt")
        if trainer.is_world_process_zero():
            for key, value in results.items():
                    metrics[key] = value
            #with open(output_eval_file, "w") as writer:
            with open(output_eval_file, "a+") as writer:
                logger.info("***** Metrics results *****")
                writer.write(f"*****results *****\n")
                for key, value in metrics.items():
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

        # saving the fine tuned model & tokenizer
        #self.model.save_pretrained(model_path)
        #self.tokenizer.save_pretrained(model_path)

        print('\t>>>one iteration done!!!')
        return self.get_model_parameters(), metrics

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

def train():
    TrainConfig = TrainingArguments(
        output_dir='./results',          # output directory
        #num_train_epochs=3,              # total number of training epochs
        num_train_epochs=10,              # total number of training epochs
        per_device_train_batch_size=10,  # batch size per device during training
        per_device_eval_batch_size=20,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        #load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
        # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
        logging_steps=50,               # log & save weights each logging_steps
        #evaluation_strategy="steps",      # evaluate each `logging_steps`
        evaluation_strategy="epoch",      # evaluate each `logging_steps`
        #logging_strategy="steps",
    )

    # load the model and pass to CUDA
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(target_names)).to(device)

    trainer = Trainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=TrainConfig,                    # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=eval_dataset,          # evaluation dataset
        compute_metrics=compute_metrics,     # the callback that computes metrics of interest
    )

    # train the model
    trainer.train()

    # evaluate the current model after training
    trainer.evaluate()
    print('\t>>>evaluate the local model')

    # saving the fine tuned model & tokenizer
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

def test():
    # reload our model/tokenizer. Optional, only usable when in Python files instead of notebooks
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=len(target_names)).to(device)
    tokenizer = BertTokenizerFast.from_pretrained(model_path)

    def get_prediction(text):
        # prepare our text into tokenized sequence
        inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        # perform inference to our model
        outputs = model(**inputs)
        # get output probabilities by doing softmax
        probs = outputs[0].softmax(1)
        # executing argmax function to get the candidate label
        return target_names[probs.argmax()]

    # Example #1
    text = """
    The first thing is first.
    If you purchase a Macbook, you should not encounter performance issues that will prevent you from learning to code efficiently.
    However, in the off chance that you have to deal with a slow computer, you will need to make some adjustments.
    Having too many background apps running in the background is one of the most common causes.
    The same can be said about a lack of drive storage.
    For that, it helps if you uninstall xcode and other unnecessary applications, as well as temporary system junk like caches and old backups.
    """
    print(get_prediction(text))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_or_train', type=str,default='train',choices=['predict','train'])
    
    args = parser.parse_args()
    if args.predict_or_train == 'predict':
        test()
    else: 
        train()
        test()
