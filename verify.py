import pickle
from WebQSP.data import load_data
import torch
from tqdm import tqdm
from pprint import pprint
from collections import Counter
import numpy as np
import json
from transformers import BertTokenizer
ent2id, rel2id, triples, train_loader, val_loader = load_data('data/AnonyQA_Debug/', 'bert', 16)
ent2id1, rel2id1, triples1, train_loader1, val_loader1, test_loader = pickle.load(open('data/AnonyQA_Debug/debug_small.pkl','rb'))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

k = 0

print(tokenizer.decode(val_loader.dataset[k][1]['input_ids'][0],skip_special_tokens = True))
print(tokenizer.decode(val_loader1.dataset[k][1]['input_ids'][0],skip_special_tokens= True))
print(tokenizer.decode(val_loader.dataset[k][1]['input_ids'][0],skip_special_tokens = True) == tokenizer.decode(val_loader1.dataset[k][1]['input_ids'][0],skip_special_tokens= True))
if not torch.equal(val_loader.dataset[k][0],val_loader1.dataset[k][0]):
    print(val_loader.dataset[k][0].nonzero())
    print(val_loader1.dataset[k][0].nonzero())

print(len(ent2id))
print(len(ent2id1))

print(ent2id['Q2643'])
print(ent2id1['Q2643'])