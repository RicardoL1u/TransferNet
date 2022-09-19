import json
datapath = '../data/AnonyQA/'
files = ['train.json','eval.json','test.json']
topic_entity_set = set()
for file in files:
    for q in json.load(open(f'{datapath}{file}')):
        topic_entity_set.add(q['topic_entity'])

essentail_ttl = []
with open('../data/kg/complex_wikidata5m.ttl') as f:
    for line in f.readlines():
        l = line.strip().split('\t')
        s = l[0].strip()
        p = l[1].strip()
        o = l[2].strip()
        if s in topic_entity_set or o in topic_entity_set:
            essentail_ttl.append(line)

with open('../data/kg/essentail.ttl','w') as f:
    f.writelines(essentail_ttl)