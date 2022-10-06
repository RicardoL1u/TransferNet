import pickle
import os
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from collections import defaultdict
from utils.misc import batch_device
from .data import load_data,load_data_for_anonyqa
from .model import TransferNet

from IPython import embed


def validate(args, model, data, device, name='val', verbose = False):
    model.eval()
    count = 0
    correct = 0
    hop_count = defaultdict(list)
    e_score_list = []
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            outputs = model(*batch_device(batch, device)) # [bsz, Esize]
            e_score = outputs['e_score'].cpu()
            e_score_list.append(e_score)
            scores, idx = torch.max(e_score, dim = 1) # [bsz], [bsz]
            match_score = torch.gather(batch[2], 1, idx.unsqueeze(-1)).squeeze().tolist()
            count += len(match_score)
            correct += sum(match_score)
            for i in range(len(match_score)):
                # 对于每一个答案，用argmax找出他是在哪一hop找出的答案
                h = outputs['hop_attn'][i].argmax().item()
                hop_count[h].append(match_score[i])

            if verbose:
                answers = batch[2]
                for i in range(len(match_score)):
                    if match_score[i] == 0:
                        print('================================================================')
                        question_ids = batch[1]['input_ids'][i].tolist()
                        question_tokens = data.tokenizer.convert_ids_to_tokens(question_ids)
                        print(' '.join(question_tokens))
                        topic_id = batch[0][i].argmax(0).item()
                        print('> topic entity: {}'.format(data.id2ent[topic_id]))
                        for t in range(2):
                            print('>>>>>>> step {}'.format(t))
                            tmp = ' '.join(['{}: {:.3f}'.format(x, y) for x,y in 
                                zip(question_tokens, outputs['word_attns'][t][i].tolist())])
                            print('> Attention: ' + tmp)
                            print('> Relation:')
                            rel_idx = outputs['rel_probs'][t][i].gt(0.9).nonzero().squeeze(1).tolist()
                            for x in rel_idx:
                                print('  {}: {:.3f}'.format(data.id2rel[x], outputs['rel_probs'][t][i][x].item()))

                            print('> Entity: {}'.format('; '.join([data.id2ent[_] for _ in outputs['ent_probs'][t][i].gt(0.8).nonzero().squeeze(1).tolist()])))
                        print('----')
                        print('> max is {}'.format(data.id2ent[idx[i].item()]))
                        print('> golden: {}'.format('; '.join([data.id2ent[_] for _ in answers[i].gt(0.9).nonzero().squeeze(1).tolist()])))
                        print('> prediction: {}'.format('; '.join([data.id2ent[_] for _ in e_score[i].gt(0.9).nonzero().squeeze(1).tolist()])))
                        print(' '.join(question_tokens))
                        print(outputs['hop_attn'][i].tolist())
                        embed()
    acc = correct / count
    with open(f'debug_e_last_{name}.pkl','wb') as f:
        pickle.dump(e_score_list,f)
    print('pred hop accuracy: 1-hop {} (total {}), 2-hop {} (total {})'.format(
        sum(hop_count[0])/(len(hop_count[0])+0.1),
        len(hop_count[0]),
        sum(hop_count[1])/(len(hop_count[1])+0.1),
        len(hop_count[1]),
        ))
    return acc

def validate_AnonyQA(args, model, data, device,verbose = False):
    model.eval()
    total_acc = 0
    data = data.dataset 
    with torch.no_grad():
        for batch in data:
            batch = batch_device(batch, device)
            # print(batch[0].shape)
            # # print(batch[0].shape)
            # print(batch[2].shape)
            # print(batch[3].shape)
            outputs = model(
                heads=batch[0].unsqueeze(0),
                questions=batch[1],
                answers=batch[2].unsqueeze(0),
                entity_range=batch[3].unsqueeze(0)
            ) # [bsz, Esize]
            not_in_kg_num = batch[4]
            ans = batch[2]
            
            preds = (outputs['e_score'].squeeze() > 4e-1)
            # print(ans.shape)
            # print(preds.shape)
            # print(not_in_kg_num)
            matchs = torch.logical_and(preds,ans)
            # scores, idx = torch.max(e_score, dim = 1) # [bsz], [bsz]
            # match_score = torch.gather(batch[2], 1, idx.unsqueeze(-1)).squeeze().tolist()
            ans_num = ans.sum().cpu().detach().item()
            correct_num = matchs.sum().cpu().detach().item()
            pred_num = preds.sum().cpu().detach().item()
            total_ans_num = ans_num + not_in_kg_num
            p1 = total_ans_num / pred_num if pred_num > total_ans_num else  pred_num / total_ans_num
            acc = correct_num / pred_num if pred_num != 0 else 0
            total_acc += (acc*p1)
    total_acc /= len(data)
    return total_acc

def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', default = './input')
    parser.add_argument('--ckpt', required = True)
    parser.add_argument('--mode', default='val', choices=['val', 'vis', 'test'])
    parser.add_argument('--bert_name', default='bert-base-uncased', choices=['roberta-base', 'bert-base-uncased'])
    parser.add_argument('--kg_name', default='debug_small',)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if 'AnonyQA' in args.input_dir:
        ent2id, rel2id, triples, train_loader, val_loader,test_loader = load_data_for_anonyqa(args.input_dir, args.bert_name, args.kg_name,16)
    else:
        ent2id, rel2id, triples, train_loader, val_loader = load_data(args.input_dir, args.bert_name, 16)

    model = TransferNet(args, ent2id, rel2id, triples)
    missing, unexpected = model.load_state_dict(torch.load(args.ckpt), strict=False)
    if missing:
        print("Missing keys: {}".format("; ".join(missing)))
    if unexpected:
        print("Unexpected keys: {}".format("; ".join(unexpected)))
    model = model.to(device)
    # model.triples = model.triples.to(device)
    model.Msubj = model.Msubj.to(device)
    model.Mobj = model.Mobj.to(device)
    model.Mrel = model.Mrel.to(device)

    if args.mode == 'vis':
        validate(args, model, val_loader, device, True)
    elif args.mode == 'val':
        validate(args, model, val_loader, device, name='val', verbose =  False)
        validate(args, model, test_loader, device, name='test', verbose = False)

if __name__ == '__main__':
    main()
