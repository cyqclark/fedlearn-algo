#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

is_visual = True

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

def read_20newsgroups(test_size=0.2):
      # download & load 20newsgroups dataset from sklearn's repos
      dataset = fetch_20newsgroups(subset="all", shuffle=True, remove=("headers", "footers", "quotes"))
      documents = dataset.data
      labels = dataset.target
      # split into training & testing a return data as well as label names
      return train_test_split(documents, labels, test_size=test_size), dataset.target_names

def twenty_newsgroup_to_csv():
    #newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    #newsgroups = fetch_20newsgroups(subset="all", shuffle=True, remove=("headers", "footers", "quotes"))
    newsgroups = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    #newsgroups = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    #newsgroups = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))

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
        print(dict_users[i].head(), dict_users[i].shape)

        if is_visual:
            visulize_distribution(dict_users[i])
    #print(dict_users.keys())
    return dict_users

def noniid_20newsgroups(dataset, num_users):
    """
    Sample non-I.I.D client data from 20newsgroups dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


if __name__ == '__main__':
    if 0:
        (train_texts, valid_texts, train_labels, valid_labels), target_names = read_20newsgroups()
        print(len(train_texts), len(train_labels))
        print(len(valid_texts), len(valid_labels))
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
    dataset = twenty_newsgroup_to_csv()
    #print(dataset.head(10))
    #dataset = fetch_20newsgroups(subset="all", shuffle=True, remove=("headers", "footers", "quotes"))
    iid_20newsgroups(dataset, 2)

