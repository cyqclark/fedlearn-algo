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

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

is_visual = True
is_to_csv = True #False
def visulize_distribution(df):
    if 1:
        print(df.target.value_counts())
        #df.target.value_counts()
    else:
        import matplotlib.pyplot as plt
        print('++')
        df['target'].plot.hist(width=0.1, )
        #plt.hist(column='target')
        #plt.hist(out['target'])
        print('--')
        plt.show()

def read_20newsgroups(data_file=None, test_file=None, dataset=None, test_size=0.2):
    if test_file is not None:
        testset = pd.read_csv(test_file)
        testset = testset.dropna()
        if is_visual:
            visulize_distribution(testset)
        valid_texts = list(testset['text'])
        valid_labels = np.array(testset['target'])
        classifier_types = list(testset['title'].unique())

        dataset = pd.read_csv(data_file)
        dataset = dataset.dropna()
        train_texts = list(dataset['text'])
        train_labels = np.array(dataset['target'])
        classifier_types = list(dataset['title'].unique())
        if is_visual:
            visulize_distribution(dataset)

        return (train_texts, valid_texts, train_labels, valid_labels), classifier_types
    else: 
        if data_file is not None:
            print(data_file)
            dataset = pd.read_csv(data_file)
            #https://stackoverflow.com/questions/63517293/valueerror-textencodeinput-must-be-uniontextinputsequence-tupleinputsequence
            dataset = dataset.dropna()
            #print(dataset.shape)
        
        if dataset is not None: 
            #print(dataset.shape)
            #print(dataset.columns)
            documents = list(dataset['text'])
            labels = np.array(dataset['target'])
            classifier_types = list(dataset['title'].unique())
            #print(type(documents), len(documents), documents[0])
            #print(type(labels), len(labels), labels[0])
            #print(classifier_types, len(classifier_types))
        else:
            # download & load 20newsgroups dataset from sklearn's repos
            dataset = fetch_20newsgroups(subset="all", shuffle=True, remove=("headers", "footers", "quotes"))
            print(type(dataset))
            documents = dataset.data
            labels = dataset.target
            classifier_types = dataset.target_names
            #print(type(labels), len(labels), labels[0])
            #print(type(dataset.target_names), dataset.target_names, len(dataset.target_names))
            # split into training & testing a return data as well as label names
        print(type(documents), len(documents))
        print('>>', documents[0])
        print('>>', documents[1])
        return train_test_split(documents, labels, test_size=test_size), classifier_types

def twenty_newsgroup_to_csv(subset=None):
    #newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    #newsgroups = fetch_20newsgroups(subset="all", shuffle=True, remove=("headers", "footers", "quotes"))
    #newsgroups = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    #newsgroups = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    #newsgroups = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))
    if subset is not None:
        newsgroups = fetch_20newsgroups(subset=subset, remove=("headers", "footers", "quotes"))

    df = pd.DataFrame([newsgroups.data, newsgroups.target.tolist()]).T
    df.columns = ['text', 'target']

    targets = pd.DataFrame( newsgroups.target_names)
    targets.columns=['title']

    out = pd.merge(df, targets, left_on='target', right_index=True)
    print(out.shape, out.columns)
    #out.describe(include=['target'])
    #out.to_csv('20_newsgroup.csv')
    #out.groupby('target').count().plot.bar()
    if is_visual:
        visulize_distribution(out)
    return out

def test_20newsgroups(dataset):
    if is_to_csv:
        dataset.to_csv('test_20newsgroups.csv', index=False)

def iid_20newsgroups(dataset, num_users):
    """
    Sample I.I.D. client data from 20newsgroups dataset
    :param dataset:
    :param num_users:
    :return: dict of users' dataset
    """
    num_items = int(len(dataset)/num_users)

    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    print(dict_users, num_items)
    
    for i in range(num_users):
        chosen_idxs = np.random.choice(all_idxs, num_items, replace=False)
        dict_users[i] = dataset.iloc[chosen_idxs]
        all_idxs = list(set(all_idxs) - set(chosen_idxs))
        #print({x for i, x in enumerate(dict_users[i]) if i < 5})
        if is_visual:
            print(dict_users[i].head(), dict_users[i].shape)
            visulize_distribution(dict_users[i])

        if is_to_csv:
            dict_users[i].to_csv('iid_20newsgroups_'+str(i)+'.csv', index=False)
    #print(dict_users.keys())
    return dict_users

def noniid_label_20newsgroups(dataset, num_users, alpha=None):
    """
    Sample non-I.I.D client data from 20newsgroups dataset: label imbalance, quantity uniform
    :param dataset:
    :param num_users:
    :alpha: label ratio, total number = 20lables
    :return:
    """
    if is_visual:
        visulize_distribution(dataset)
    #dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    dict_users = {i: np.array([]) for i in range(num_users)}
    labels = np.array(dataset['target'])
    num_samples = len(dataset)
    num_labels = 20
    num_shards = int(len(dataset)/num_labels)
    idxs = np.arange(num_samples)
    print(dict_users)
    print(labels, len(labels))
    print(idxs, len(idxs))

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    #print(idxs_labels, len(idxs_labels))
    #idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    #print(idxs_labels)
    #idxs = idxs_labels[0, :]
    #print(idxs, len(idxs)) 

    safe_idxs = []
    seed_idxs = {}
    for i in range(len(dataset)): #only two users
        key  = idxs_labels[1][i]
        if key in seed_idxs:
            if seed_idxs[key] < 3:
                safe_idxs.append(idxs_labels[0][i]) 
            seed_idxs[key] += 1
        else:
            safe_idxs.append(idxs_labels[0][i]) 
            seed_idxs[key] = 1
        #seed_idxs[idxs_labels[1][i]] = idxs_labels[0][i]
    print('seed_idxs', seed_idxs)

    chosen_idxs = {i:[] for i in range(num_users)}
    #for i in range(18000,len(idxs)):
    #for i in range(100):
    for i in range(len(dataset)): #only two users
        user_id = idxs_labels[1][i] % 2
        if user_id == 0:
            #print(i, idxs_labels[0][i], idxs_labels[1][i])
            chosen_idxs[user_id].append(idxs_labels[0][i])
        else:
            chosen_idxs[user_id].append(idxs_labels[0][i])
    for i in range(num_users):
        dict_users[i] = dataset.iloc[chosen_idxs[i] + safe_idxs]
        #all_idxs = list(set(all_idxs) - set(chosen_idxs))
        #print({x for i, x in enumerate(dict_users[i]) if i < 5})
        if is_visual:
            print(dict_users[i].head(), dict_users[i].shape)
            visulize_distribution(dict_users[i])

        if is_to_csv:
            dict_users[i].to_csv('noniid_label_20newsgroups_alpha'+ str(alpha)+ '_'+str(i)+'.csv', index=False)

    return dict_users

def noniid_quantity_20newsgroups(dataset, num_users=2, beta=None):
    """
    Sample non-I.I.D client data from 20newsgroups dataset: quantity imbalance, label uniform
    :param dataset:
    :param num_users:
    :return:
    """
    if is_visual:
        visulize_distribution(dataset)
    #dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    num_items = {} #int(len(dataset)/num_users)
    for i in range(len(beta)):
        num_items[i] = int(len(dataset) * beta[i])

    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    print(dict_users, num_items)
    
    for i in range(num_users):
        chosen_idxs = np.random.choice(all_idxs, num_items[i], replace=False)
        dict_users[i] = dataset.iloc[chosen_idxs]
        all_idxs = list(set(all_idxs) - set(chosen_idxs))
        #print({x for i, x in enumerate(dict_users[i]) if i < 5})
        if is_visual:
            print(dict_users[i].head(), dict_users[i].shape)
            visulize_distribution(dict_users[i])

        if is_to_csv:
            dict_users[i].to_csv('noniid_quantity_20newsgroups_beta'+ str(beta[i])+ '_'+str(i)+'.csv', index=False)
    #print(dict_users.keys())
    return dict_users



if __name__ == '__main__':
    if 0:
        (train_texts, valid_texts, train_labels, valid_labels), target_names = read_20newsgroups()
        print(type(train_texts), len(train_texts))
        print(type(train_labels), len(train_labels))
        if 0:
            start=0
            valid_sample_n = 2
            sample_n = valid_sample_n*5
            train_texts = train_texts[start:sample_n]
            train_labels = train_labels[start:sample_n]
            valid_texts = valid_texts[start:valid_sample_n]
            valid_labels = valid_labels[start:valid_sample_n]
            print(len(train_texts), len(train_labels))
            print(len(valid_texts), len(valid_labels))
            #print(valid_texts, valid_labels)
        print(target_names)
    if 0: #generate iid-dataset
        dataset = twenty_newsgroup_to_csv()
        #print(dataset.head(10))
        #dataset = fetch_20newsgroups(subset="all", shuffle=True, remove=("headers", "footers", "quotes"))
        dict_user = iid_20newsgroups(dataset, 2)
        read_20newsgroups(dict_user[0])
        read_20newsgroups()
    if 0: #load dataset via read_20newsgroups
        #(train_texts, valid_texts, train_labels, valid_labels), target_names = read_20newsgroups(data_file=None)
        #(train_texts, valid_texts, train_labels, valid_labels), target_names = read_20newsgroups(data_file='iid_20newsgroups_1.csv')
        (train_texts, valid_texts, train_labels, valid_labels), target_names = read_20newsgroups(data_file='noniid_label_20newsgroups_alpha0.5_0.csv', test_file='test_20newsgroups.csv')
        print(type(train_texts), len(train_texts))
        print(type(train_labels), len(train_labels))
        print(train_labels[:2])
    if 1:
        dataset = twenty_newsgroup_to_csv(subset='train')
        #print(dataset.head(10))
        #dataset = fetch_20newsgroups(subset="all", shuffle=True, remove=("headers", "footers", "quotes"))
        #dict_user = noniid_20newsgroups(dataset, 2)
        noniid_label_20newsgroups(dataset, 2, alpha=0.5)
        num_users = 2
        #noniid_quantity_20newsgroups(dataset, beta=[0.1, 0.9])
    if 0:
        dataset = twenty_newsgroup_to_csv(subset='test')
        test_20newsgroups(dataset)
