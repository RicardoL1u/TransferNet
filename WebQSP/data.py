import torch
import os
import pickle
from collections import defaultdict
from transformers import AutoTokenizer
from utils.misc import invert_dict
import json
from copy import deepcopy
from tqdm import tqdm
def collate(batch):
    batch = list(zip(*batch))
    topic_entity, question, answer, entity_range = batch
    topic_entity = torch.stack(topic_entity)
    question = {k:torch.cat([q[k] for q in question], dim=0) for k in question[0]}
    answer = torch.stack(answer)
    entity_range = torch.stack(entity_range)
    return topic_entity, question, answer, entity_range


class Dataset(torch.utils.data.Dataset):
    def __init__(self, questions, ent2id):
        self.questions = questions
        self.ent2id = ent2id

    def __getitem__(self, index):
        topic_entity, question, answer, entity_range = self.questions[index]
        topic_entity = self.toOneHot(topic_entity)
        answer = self.toOneHot(answer)
        entity_range = self.toOneHot(entity_range)
        return topic_entity, question, answer, entity_range

    def __len__(self):
        return len(self.questions)

    def toOneHot(self, indices):
        indices = torch.LongTensor(indices)
        vec_len = len(self.ent2id)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return one_hot

class AnonyQADataloader(torch.utils.data.DataLoader):
    def __init__(self,dataset_path,tokenizer,ent2id,rel2id,sub_map,batch_size,training=False):
        self.tokenizer = tokenizer  
        self.ent2id = ent2id       
        data = []
        
        for question in tqdm(json.load(open(dataset_path))):
            head = [self.ent2id[question['topic_entity']]]

            entity_range = set()
            for p, o in sub_map[question['topic_entity']]:
                entity_range.add(o)
                for p2, o2 in sub_map[o]:
                    entity_range.add(o2)
            entity_range = [ent2id[o] for o in entity_range]
            assert entity_range != [],print(question['topic_entity'],sub_map[question['topic_entity']])

            tokenized_q = self.tokenizer(question['text'].strip(),question['question'].strip(), max_length=512, padding='max_length', return_tensors="pt",truncation='only_first')
            # if len(tokenized_q['input_ids']) > 512:
            #     print(question['text'].strip(),question['question'].strip())
            #     for k,v in tokenized_q.items():
            #         print(k,v.shape)
            # tokenized_q = self.tokenizer(question['text'].strip() + ' <spt> ' + question['question'].strip(), max_length=512, padding='max_length', return_tensors="pt")
            ans = [ent2id[a] for a in question['ans_ids'] if a in ent2id.keys()]
            if len(ans) == 0:
                continue
            data.append([head, tokenized_q, ans, entity_range])

        print('data number: {}'.format(len(data)))
        
        dataset = Dataset(data, ent2id)

        super().__init__(
            dataset, 
            batch_size=batch_size,
            shuffle=training,
            collate_fn=collate, 
            )
        # super().__init__()

class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, input_dir, fn, bert_name, ent2id, rel2id, batch_size, training=False):
        print('Reading questions from {}'.format(fn))
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name)
        self.ent2id = ent2id
        self.rel2id = rel2id
        self.id2ent = invert_dict(ent2id)
        self.id2rel = invert_dict(rel2id)


        # The functionality of both dictionaries and defaultdict are almost same 
        # except for the fact that defaultdict never raises a KeyError. 
        # It provides a default value for the key that does not exists.
        sub_map = defaultdict(list)
        so_map = defaultdict(list)
        for line in open(os.path.join(input_dir, 'fbwq_full/train.txt')):
            l = line.strip().split('\t')
            s = l[0].strip()
            p = l[1].strip()
            o = l[2].strip()
            sub_map[s].append((p, o))
            so_map[(s, o)].append(p)


        data = []
        for line in open(fn):
            line = line.strip()
            if line == '':
                continue
            line = line.split('\t')
            # if no answer
            if len(line) != 2:
                continue
            question = line[0].split('[')
            question_1 = question[0]
            question_2 = question[1].split(']')
            head = question_2[0].strip()
            question_2 = question_2[1]
            # question = question_1 + 'NE' + question_2
            question = question_1.strip()
            ans = line[1].split('|')


            # if (head, ans[0]) not in so_map:
            #     continue

            entity_range = set()
            for p, o in sub_map[head]:
                entity_range.add(o)
                for p2, o2 in sub_map[o]:
                    entity_range.add(o2)
            entity_range = [ent2id[o] for o in entity_range]

            head = [ent2id[head]]
            question = self.tokenizer(question.strip(), max_length=64, padding='max_length', return_tensors="pt")
            ans = [ent2id[a] for a in ans]
            data.append([head, question, ans, entity_range])

        print('data number: {}'.format(len(data)))
        
        dataset = Dataset(data, ent2id)

        super().__init__(
            dataset, 
            batch_size=batch_size,
            shuffle=training,
            collate_fn=collate, 
            )


def load_data_for_anonyqa(datapath,bert_name,kg_name,batch_size):
    cache_fn = os.path.join(datapath,f'{kg_name}.pkl')
    if os.path.exists(cache_fn):
        print('Read from cache file: {} (NOTE: delete it if you modified data loading process)'.format(cache_fn))
        with open(cache_fn, 'rb') as fp:
            ent2id, rel2id, triples, train_dataloader, valid_dataloader = pickle.load(fp)
        print('Train number: {}, test number: {}'.format(len(train_dataloader.dataset), len(valid_dataloader.dataset)))
    else:
        files = ['train.json','eval.json','test.json']
        topic_entity_set = set()
        for file in files:
            for q in json.load(open(f'{datapath}/{file}')):
                topic_entity_set.add(q['topic_entity'])
                
        sub_map = defaultdict(list)
        so_map = defaultdict(list)
        triples = []
        entity_set = set()
        relation_set = set()        
        for line in set(open(f'data/kg/{kg_name}.ttl').readlines()).union(set(open('data/kg/essentail.ttl').readlines())):
            l = line.strip().split('\t')
            s = l[0].strip()
            p = l[1].strip()
            o = l[2].strip()
            entity_set.add(s)
            entity_set.add(o)
            relation_set.add(p)
        
        ent2id = {}
        rel2id = {}
        for entity in entity_set:
            ent2id[entity] = len(ent2id)
        for relation in relation_set:
            rel2id[relation] = len(rel2id)
            rel2id[relation+'^{-1}'] = len(rel2id)
        print(f'entity number is {len(ent2id)}')
        print(f'relation number is {len(rel2id)}')
        for line in set(open(f'data/kg/{kg_name}.ttl').readlines()).union(set(open('data/kg/essentail.ttl').readlines())):
            l = line.strip().split('\t')
            s = l[0].strip()
            p = l[1].strip()
            o = l[2].strip()
            sub_map[s].append((p, o))
            if o in topic_entity_set:
                sub_map[o].append((p+'^{-1}',s))
            so_map[(s, o)].append(p)
            triples.append((ent2id[s], rel2id[p], ent2id[o]))
            # p_rev = rel2id[l[1].strip()+'^{-1}']
            triples.append((ent2id[o], rel2id[l[1].strip()+'^{-1}'], ent2id[s]))
        triples = torch.LongTensor(triples)
        # print(f'the number of missed q is {len(missied_q)}')
        tokenizer = AutoTokenizer.from_pretrained(bert_name)
        # special_tokens = ['<spt>']
        # print(f'add speciall tokens {special_tokens}')
        # tokenizer.add_tokens(special_tokens,special_tokens=True)

        train_dataloader = AnonyQADataloader(os.path.join(datapath,'train.json'),tokenizer,ent2id,rel2id,sub_map,batch_size,True)
        valid_dataloader = AnonyQADataloader(os.path.join(datapath,'eval.json'),tokenizer,ent2id,rel2id,sub_map,batch_size,False)

        with open(cache_fn, 'wb') as fp:
            pickle.dump((ent2id, rel2id, triples, train_dataloader, valid_dataloader), fp)

    return ent2id, rel2id, triples, train_dataloader, valid_dataloader


def load_data(input_dir, bert_name, batch_size):
    cache_fn = os.path.join(input_dir, 'processed.pt')
    if os.path.exists(cache_fn):
        print('Read from cache file: {} (NOTE: delete it if you modified data loading process)'.format(cache_fn))
        with open(cache_fn, 'rb') as fp:
            ent2id, rel2id, triples, train_data, test_data = pickle.load(fp)
        print('Train number: {}, test number: {}'.format(len(train_data.dataset), len(test_data.dataset)))
    else:
        print('Read data...')
        ent2id = {}
        for line in open(os.path.join(input_dir, 'fbwq_full/entities.dict')):
            l = line.strip().split('\t')
            ent2id[l[0].strip()] = len(ent2id) # one usefull increment trick
        # print(len(ent2id))
        # print(max(ent2id.values()))
        rel2id = {}
        for line in open(os.path.join(input_dir, 'fbwq_full/relations.dict')):
            l = line.strip().split('\t')
            rel2id[l[0].strip()] = int(l[1])

        triples = []
        for line in open(os.path.join(input_dir, 'fbwq_full/train.txt')):
            l = line.strip().split('\t')
            s = ent2id[l[0].strip()]
            p = rel2id[l[1].strip()]
            o = ent2id[l[2].strip()]
            triples.append((s, p, o))
            p_rev = rel2id[l[1].strip()+'_reverse']
            triples.append((o, p_rev, s))
        triples = torch.LongTensor(triples)

        train_data = DataLoader(input_dir, os.path.join(input_dir, 'QA_data/WebQuestionsSP/qa_train_webqsp.txt'), bert_name, ent2id, rel2id, batch_size, training=True)
        test_data = DataLoader(input_dir, os.path.join(input_dir, 'QA_data/WebQuestionsSP/qa_test_webqsp.txt'), bert_name, ent2id, rel2id, batch_size)
    
        with open(cache_fn, 'wb') as fp:
            pickle.dump((ent2id, rel2id, triples, train_data, test_data), fp)

    return ent2id, rel2id, triples, train_data, test_data
