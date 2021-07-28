import argparse
from dataclasses import dataclass, field
import os
from typing import Optional, Union,Dict,List,Any
from collections import OrderedDict

import torch
import transformers
#from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers.file_utils import is_torch_available
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

model_path = "20newsgroups-bert-base-uncased"
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
            print('is pytorch...')
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # ^^ safe to call this function even if cuda is not available

    set_seed(1)

    # the model we gonna train, base uncased BERT
    # check text classification models here: https://huggingface.co/models?filter=text-classification
    model_name = "bert-base-uncased"
    # max sequence length for each document/sentence sample
    max_length = 512

    # load the tokenizer
    tokenizer = transformers.BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)

    def read_20newsgroups(test_size=0.2):
      # download & load 20newsgroups dataset from sklearn's repos
      dataset = fetch_20newsgroups(subset="all", shuffle=True, remove=("headers", "footers", "quotes"))
      documents = dataset.data
      labels = dataset.target
      # split into training & testing a return data as well as label names
      return train_test_split(documents, labels, test_size=test_size), dataset.target_names

    # call the function
    (train_texts, valid_texts, train_labels, valid_labels), target_names = read_20newsgroups()
    print(len(train_texts), len(train_labels))
    print(len(valid_texts), len(valid_labels))
    if 0:
        train_texts = train_texts[:80]
        valid_texts = valid_texts[:80]
        train_labels = train_labels[:20]
        valid_labels = valid_texts[:20]
        print(len(train_texts), len(train_labels))
        print(len(valid_texts), len(valid_labels))
    print(target_names)

    # tokenize the dataset, truncate when passed `max_length`,
    # and pad with 0's when less than `max_length`
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
    valid_dataset = NewsGroupsDataset(valid_encodings, valid_labels)

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

TrainConfig = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,              # total number of training epochs
        #num_train_epochs=1,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=20,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        #load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
        # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
        logging_steps=200,               # log & save weights each logging_steps
        #evaluation_strategy="steps"      # evaluate each `logging_steps`
    )


from sklearn.metrics import accuracy_score
def compute_metrics(pred):
      labels = pred.label_ids
      preds = pred.predictions.argmax(-1)
      # calculate accuracy using sklearn's function
      acc = accuracy_score(labels, preds)
      return {
          'accuracy': acc,
      }

class BertTextClassifier():
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

        # load the model and pass to CUDA
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(target_names)).to("cuda")
        # load the tokenizer
        self.tokenizer = transformers.BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)

    def train(self):
        print('\t========BertTextClassifier=========', )

        trainer = Trainer(
            model=self.model,                         # the instantiated Transformers model to be trained
            args=self.train_config,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=valid_dataset,          # evaluation dataset
            compute_metrics=compute_metrics,     # the callback that computes metrics of interest
        )

        # train the model
        trainer.train()

        # evaluate the current model after training
        trainer.evaluate()

        # saving the fine tuned model & tokenizer
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)

        return_matrics = {}
        return self.get_model_parameters(), return_matrics

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
    # load the model and pass to CUDA
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(target_names)).to("cuda")

    trainer = Trainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=TrainConfig,                    # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=valid_dataset,          # evaluation dataset
        compute_metrics=compute_metrics,     # the callback that computes metrics of interest
    )

    # train the model
    trainer.train()

    # evaluate the current model after training
    trainer.evaluate()

    # saving the fine tuned model & tokenizer
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

def test():
    # reload our model/tokenizer. Optional, only usable when in Python files instead of notebooks
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=len(target_names)).to("cuda")
    tokenizer = BertTokenizerFast.from_pretrained(model_path)

    def get_prediction(text):
        # prepare our text into tokenized sequence
        inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
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

    # Example #2
    text = """
    A black hole is a place in space where gravity pulls so much that even light can not get out.
    The gravity is so strong because matter has been squeezed into a tiny space. This can happen when a star is dying.
    Because no light can get out, people can't see black holes.
    They are invisible. Space telescopes with special tools can help find black holes.
    The special tools can see how stars that are very close to black holes act differently than other stars.
    """
    print(get_prediction(text))

    # Example #3
    text = """
    Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus.
    Most people infected with the COVID-19 virus will experience mild to moderate respiratory illness and recover without requiring special treatment.
    Older people, and those with underlying medical problems like cardiovascular disease, diabetes, chronic respiratory disease, and cancer are more likely to develop serious illness.
    """
    print(get_prediction(text))

    #target_names
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_or_train', type=str,default='train',choices=['predict','train'])
    
    args = parser.parse_args()
    if args.predict_or_train == 'predict':
        test()
    else: 
        train()
        test()
