import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.datasets import fetch_20newsgroups
import os
import numpy as np
from scipy import stats
import json
from datasets import load_dataset

DATA = "./data"
TEST_BATCH = 16

'''
Modify for centralized training rather than federated training
'''
def build_dataset(dataset, batch_size, seed=0):
    if dataset == '20newsgroups':
        trainset, testset = build_20newsgroups()
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        testloader = DataLoader(testset, batch_size=TEST_BATCH, shuffle=False, num_workers=0)
    elif dataset == 'openwebtext':
        trainset = build_openwebtext()
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        testloader = None
    # clientloaders = [DataLoader(client, batch_size=batch_size, shuffle=True, num_workers=0) for client in clients]
    # trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    # testloader = DataLoader(testset, batch_size=TEST_BATCH, shuffle=False, num_workers=0)
    return trainloader, testloader

def build_openwebtext():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token_id = 50256

    dataset = load_dataset('openwebtext', cache_dir=f'{DATA}/openwebtext/text')

    def tokenize_function(examples):
        return tokenizer(examples['text'], max_length=128, padding='max_length', truncation=True)

    tokenized_data = dataset.map(tokenize_function, batched=True, remove_columns=["text"], cache_file_names={'train':f'{DATA}/openwebtext/tokenized'}, load_from_cache_file=True)

    tokenized_data['train'].set_format(type="torch", columns=['input_ids', 'attention_mask'])
    tr_X = tokenized_data['train']['input_ids']
    tr_Y = [0 for i in range(len(tr_X))]

    trainset = list(zip(tr_X, tr_Y))
    return trainset

def build_20newsgroups():
    train_pt = f"{DATA}/20newsgroups/20newsgroups_train.pt"
    test_pt = f"{DATA}/20newsgroups/20newsgroups_test.pt"
    if not os.path.exists(train_pt) or not os.path.exists(test_pt):
        generate_20newsgroups_dump()
    tr_d = torch.load(train_pt)
    ev_d = torch.load(test_pt)
    N = len(tr_d['Y'])
    Y = tr_d['Y']
    trainset = list(zip(tr_d['X'], tr_d['Y']))
    testset = list(zip(ev_d['X'], ev_d['Y']))
    n_classes = 20
    # clients = partition_trainset(trainset, Y, n_classes, n_clients, alpha, seed)
    return trainset, testset

def generate_20newsgroups_dump():
    print("Generating 20newsgroups cache...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token_id = 50256
    ng_train = fetch_20newsgroups(subset='train')
    tr_X = torch.LongTensor([tokenizer.encode(x, max_length=128, padding='max_length', truncation=True) for x in ng_train['data']])

    ng_test = fetch_20newsgroups(subset='test')
    ev_X = torch.LongTensor([tokenizer.encode(x, max_length=128, padding='max_length', truncation=True) for x in ng_test['data']])

    tr_Y = torch.LongTensor(ng_train['target'])
    ev_Y = torch.LongTensor(ng_test['target'])

    os.makedirs(f"{DATA}/20newsgroups", exist_ok=True)
    torch.save({'X': tr_X, 'Y': tr_Y}, f"{DATA}/20newsgroups/20newsgroups_train.pt")
    torch.save({'X': ev_X, 'Y': ev_Y}, f"{DATA}/20newsgroups/20newsgroups_test.pt")
