#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import re
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BasicTokenizer
import os

basicTokenizer = BasicTokenizer()

MAX_SENT_LENGTH = 200

# 加载词典 pre-trained model tokenizer (vocabulary)
VOCAB = './bert-large-cased-vocab.txt'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def match_brackets(s, i):
    if s[i] == '(':
        end_bracket = ')'
    elif s[i] == '[':
        end_bracket = ']'
    elif s[i] == '{':
        end_bracket = '}'
    elif s[i] == '<':
        end_bracket = '>'   # Introduce this for start-end match
    else:
        raise ValueError('Not match')
    i += 1
    while True:
        if s[i] == end_bracket:
            return i + 1
        else:
            i = match_brackets(s, i)

def is_match(s):
    try:
        s = ''.join(c for c in s if c in ('(','[','{','}',']',')'))
        match_brackets('<' + s + '>', 0)
    except ValueError:
        return False
    else:
        return True

# {"originalText": "印度海军近日从“加尔各答”级驱逐舰上试射了由印度和以色列联合研制的远程面空导弹?试验在位于印度西海岸的“加尔各答”级驱逐舰上进行?试验中,该导弹成功拦截了一个增程型空中目标?此次试验的“巴拉克”-8导弹由印度国防研究与发展局?以色列航空工业公司?以色列武器研发与技术基础设施管理局?埃尔塔系统公司?拉法尔公司等机构联合研制?",
#  "entities": [{"label_type": "试验要素", "overlap": 0, "start_pos": 8, "end_pos": 17}, {"label_type": "试验要素", "overlap": 0, "start_pos": 34, "end_pos": 39}, {"label_type": "试验要素", "overlap": 0, "start_pos": 80, "end_pos": 86}, {"label_type": "试验要素", "overlap": 0, "start_pos": 93, "end_pos": 101}]}

# {"entity": "bimagrumab", "type": "Drug", "start": 35, "end": 45}

def generate_train_json(data_sign):
    if data_sign == "train":
        file_path = '../origin_data/new_train.json'
    elif data_sign == "val" or data_sign == "val_test":
        file_path = '../origin_data/new_val.json'
    else:
        return None
    fr = open(file_path, 'r', encoding='utf-8')
    datas = []
    entity_type = set()
    index = 0
    for line in fr:
        index += 1
        # if index > 100:
        #     break
        line = json.loads(line)
        if data_sign == 'train':
            text = line['text'].replace('\t', '.')
            entities = line['entities']
        if data_sign == 'val':
            text = line['text'].replace('\t', '.')
            entities = []
        else:
            text = line['text']
            entities = []

        context_list = text.split('. ')
        token_lens = [len(tokenizer.tokenize(contexts)) for contexts in context_list]

        index = 0
        while index < len(token_lens):
            while index + 1 < len(token_lens) and token_lens[index] + token_lens[index + 1] < MAX_SENT_LENGTH:
                token_lens[index] += token_lens[index + 1]
                del token_lens[index + 1]
                context_list[index] = context_list[index] + '. ' + context_list[index + 1]
                del context_list[index + 1]
            index += 1

        entity_dict = {}
        label = {
            "Disease": [],
            "Phenotype": [],
            "Drug": [],
            "Organization": [],
            "Gene": [],
            "Virus": [],
            "ChemicalCompound": [],
            "Chemical": [],
        }
        label_entity = {
            "Disease": [],
            "Phenotype": [],
            "Drug": [],
            "Organization": [],
            "Gene": [],
            "Virus": [],
            "ChemicalCompound": [],
            "Chemical": [],
        }
        label_entity_ = {
            "Disease": [],
            "Phenotype": [],
            "Drug": [],
            "Organization": [],
            "Gene": [],
            "Virus": [],
            "ChemicalCompound": [],
            "Chemical": [],
        }
        label_entity__ = {
            "Disease": [],
            "Phenotype": [],
            "Drug": [],
            "Organization": [],
            "Gene": [],
            "Virus": [],
            "ChemicalCompound": [],
            "Chemical": [],
        }
        # 使用空格切分
        tokenized_text = basicTokenizer.tokenize(text)
        entity_str_list = []
        for i in entities:
            # entity_dict['label_type'] = i['type']
            # entity_dict['overlap'] = 0
            # entity_dict['start_pos'] = i['start']
            # entity_dict['end_pos'] = i['end']
            entity_type.add(i['type'])

            entity_str = text[i['start']: i['end']]
            entity_list = basicTokenizer.tokenize(entity_str)
            start_text = basicTokenizer.tokenize(text[: i['start']])
            start = len(start_text)
            end = start + len(entity_list)

            label[i['type']].append(str(start) + ',' + str(end))
            label_entity[i['type']].append(text[i['start']: i['end']])
            label_entity_[i['type']].append(tokenized_text[start: end])

            entity_str_list.append(entity_str)
        datas.append(
            {
                # "entity_str_list": entity_str_list,
                'context': text,
                # 'tokenized_text': tokenized_text,
                'label': label,
                'label_entity': label_entity,
                # 'label_entity_': label_entity_,
            }
        )
    return datas


def generate_process_data():
    datas_train = generate_train_json("train")
    # i = int(len(datas_train) / 2)
    # datas_train = datas_train[:i]
    # datas_val = generate_train_json("val")
    # i = int(len(datas_val) / 3)
    # datas_val = datas_val[6000:8000]
    # datas_val = generate_train_json("val_test")

    with open('train.json', 'w', encoding='utf-8') as file_obj:
        json.dump(datas_train, file_obj, ensure_ascii=False)
    # with open('val.json', 'w', encoding='utf-8') as file_obj:
    #     json.dump(datas_val, file_obj, ensure_ascii=False)
    # with open('val_test.json', 'w', encoding='utf-8') as file_obj:
    #     json.dump(datas_val, file_obj, ensure_ascii=False)
    print('保存成功')


def caculate_vocab_num():
    fr = open('../origin_data/new_train.json', 'r', encoding='utf-8')

    step = 0
    sets = set()
    lists = []

    for line in fr:
        # step += 1
        # if step == 50:
        #     break
        line = json.loads(line)
        i = line['text']
        ins = i.split('\t')
        strs1 = []
        for j in ins:
            strs1 = j.split(' ')
            for z in strs1:
                if z.replace('.', '').replace(',', '') not in tokenizer.vocab:
                    if not (bool(re.search(r'\d', z)) or (':' in z) or ('!' in z)):
                        sets.add(z.replace('.', '').replace(',', ''))
                        lists.append(z.replace('.', '').replace(',', ''))

    list_dict = {}
    for i in lists:
        if i not in list_dict.keys():
            list_dict[i] = 1
        else:
            list_dict[i] = list_dict[i] + 1

    list_1 = []
    list_10 = []
    list_50 = []
    list_100 = []
    list_1000 = []

    for key, value in list_dict.items():
        if value < 10:
            list_1.append(key)
        elif 10 <= value < 50:
            list_10.append(key)
        elif 50 <= value < 100:
            list_50.append(key)
        elif 100 <= value < 1000:
            list_100.append(key)
        else:
            list_1000.append(key)

    print(list_dict)

    with open('train.json', 'w', encoding = 'utf-8') as file_obj:
        json.dump(i, file_obj,ensure_ascii=False, indent=4)

    print('保存成功')


def caculate_sent_num():
    fr = open('../origin_data/new_train.json', 'r', encoding='utf-8')

    lists = []

    num_100 = 0
    num_200 = 0
    num_300 = 0
    num_400 = 0
    num_500 = 0

    for line in fr:
        line = json.loads(line)
        i = line['text']
        ins_len = re.split(r"[ \t]", i)

        ins = i.split('\t')
        ins_num = len(ins)

        lists.append({'text': i,
                      'ins': ins,
                      'ins_len': ins_len,
                      'ins_num': ins_num
                      })

        if 200 > len(ins_len) > 100:
            num_100 += 1
        if 300 > len(ins_len) > 200:
            num_200 += 1
        if 400 > len(ins_len) > 300:
            num_300 += 1
        if 500 > len(ins_len) > 400:
            num_400 += 1
        if len(ins_len) > 256:
            num_500 += 1

    print(num_100)
    print(num_200)
    print(num_300)
    print(num_400)
    print(num_500)
    print('stop')

def caculate_sent_num_for_bert():
    fr = open('../origin_data/new_train.json', 'r', encoding='utf-8')

    lists = []

    num_10 = 0
    num_100 = 0
    num_200 = 0
    num_300 = 0
    num_400 = 0
    num_500 = 0

    for line in fr:
        line = json.loads(line)
        i = line['text']
        if "\t" in i:
            print("stop")
        ins_len = tokenizer.tokenize(i)

        if 100 > len(ins_len) > 0:
            num_10 += 1
        if 200 > len(ins_len) > 100:
            num_100 += 1
        if 300 > len(ins_len) > 200:
            num_200 += 1
        if 400 > len(ins_len) > 300:
            num_300 += 1
        if 500 > len(ins_len) > 400:
            num_400 += 1
        if len(ins_len) > 500:
            num_500 += 1

    print(num_10)
    print(num_100)
    print(num_200)
    print(num_300)
    print(num_400)
    print(num_500)
    print('stop')

def generate_result():
    fr1 = open('result.json', 'r', encoding='utf-8')
    span_triple_lst = json.load(fr1)
    fr2 = open('val_test.json', 'r', encoding='utf-8')
    lines = json.load(fr2)

    # span_triple_lst = span_triple_lst[:80]
    # lines = lines[:10]

    sent_list = []
    for line in lines:
        for i in range(8):
            # sent_list.append({'context': line['context'], 'entity': line['label'], 'entity_str': line['label_entity']})
            sent_list.append({'context': line['context'], 'entity': line['label']})
    result = []
    submit_entity_prob = []
    submit = []
    sents = []

    index_sent = 0

    print(len(sent_list))
    print(len(span_triple_lst))

    for sent, triple in zip(sent_list, span_triple_lst):
        index_sent += 1
        # if index_sent > 1000:
        #     break
        # 获取真实实体
        if len(triple) is 0:
            continue
        tag = triple[0]['tag']
        query_len = type_len[tag]

        # 获取预测的实体
        bert_ins = tokenizer.tokenize(sent['context'])

        entity_prob = []
        # entity_nest = []

        # ！！ 使用规则的方式匹配实体
        # 第一步，首先将被切分成多个片段的实体组合成一个完整的实体片段
        # for i in triple:
        #     # 使用规则删除重叠实体
        #     # if i['begin'] in entity_nest:
        #     #     continue
        #     # 使用规则删除过长的片段
        #     # if i['end'] - i['begin'] > 40:
        #     #     continue
        #     # entity_nest.append(i['begin'])
        #
        #     entity = bert_ins[i['begin'] - query_len - 2: i['end'] - query_len - 2]
        #     span_true = []
        #     flag = False
        #
        #     for index in range(len(entity)):
        #         if index == 0:
        #             span_true.append(entity[index])
        #         elif "##" in entity[index]:
        #             # 带有 ## 的片段和前面的片段进行拼接
        #             span_true.append(span_true.pop() + entity[index].replace("##", ''))
        #         elif "-" is entity[index]:
        #             # 带有 - 的片段和前后两个片段进行组合
        #             span_true.append(span_true.pop() + entity[index] + entity[index + 1])
        #             flag = True
        #         elif flag:
        #             flag = False
        #         else:
        #             span_true.append(entity[index])
        #     span_true = ' '.join(span_true)
        #
        #     # 解决括号不匹配问题
        #     # if span_true.count("(") != span_true.count(")"):
        #     #     continue
        #     if not is_match(span_true):
        #         continue
        #
        #     entity_prob.append(span_true)

        # !! 使用正则表达式的方式匹配实体
        for i in triple:
            entity = bert_ins[i['begin'] - query_len - 2: i['end'] - query_len - 2]

            entity_span_list = []
            for spans in entity:
                entity_span_list.append(
                    spans.replace(')', "\)").replace('(', "\(") \
                    .replace('$', "\$").replace('?', "\?").replace('*', "\*") \
                    .replace('+', "\+").replace('.', "\.").replace('[', "\[") \
                    .replace('{', "\{").replace('|', "\|").replace('^', "\^") \
                    .replace('##', "")
                )
            span_true = '\s*'.join(entity_span_list)

            # 解决括号不匹配问题
            if not is_match(span_true):
                continue

            pattern = re.compile(span_true)

            entity_prob.append(pattern)
        # result.append({'context': sent['context'], 'tag': tag, 'entity_truth': sent['entity_str'][tag], 'entity_prob': entity_prob})
        result.append({'context': sent['context'], 'tag': tag, 'entity_prob': entity_prob})

    # with open('results.json', 'w', encoding = 'utf-8') as file_obj:
    #     json.dump(result, file_obj,ensure_ascii=False, indent=4)
    # print(len(result))

    sents = []
    submit_entity_prob = []
    # 第二部，按标准答案的要求，修改定位实体位置的标签
    for result_ in result:

        # 首先拿到所需内容
        ins = result_['context']
        tag = result_['tag']
        entity_prob = result_['entity_prob']

        ins_lower = ins.lower()

        remeber = []
        # 将原始句子中的词全部变成小写，然后查找实体在句子中的位置
        for entity in entity_prob:
            # 去掉重复片段
            if entity in remeber:
                break
            remeber.append(entity)
            for i in re.finditer(entity, ins_lower):
                start = i.start()
                end = i.end()
                # {"entity": "tetanus", "type": "Disease", "start": 14, "end": 21}
                submit_entity_prob.append(
                    {"entity": ins[start: end], "type": tag, "start": start, "end": end})

        # if len(sents) == 0:
        #     sents.append(ins)
        #     submit.append({'text': ins, 'entities': submit_entity_prob})
        # if ins not in sents:
        #     if len(sents) > 0:
        #         submit.append({'text': sents.pop(), 'entities': submit_entity_prob})
        #         submit_entity_prob = []
        #     sents.append(ins)

        if ins not in sents:
            submit.append({'text': ins, 'entities': submit_entity_prob})
            sents.append(ins)
            submit_entity_prob = []
        else:
            submit[-1]['entities'] += submit_entity_prob
            submit_entity_prob = []

    with open('submit.json', 'w', encoding = 'utf-8') as file_obj:
        json.dump(submit, file_obj, ensure_ascii=False, indent=4)
    print(len(submit))


# 利用有标数据构建一个实体知识图谱
def generate_kg():
    kg_entity = set()
    file_path = '../origin_data/new_train.json'
    fr = open(file_path, 'r', encoding='utf-8')

    for line in fr:
        line = json.loads(line)
        entities = line['entities']
        for entity in entities:
            kg_entity.add((entity['entity'], entity['type']))

    with open('kg.json', 'w', encoding = 'utf-8') as file_obj:
        json.dump(list(kg_entity), file_obj, ensure_ascii=False, indent=4)


# 将知识图谱对齐测试集得到一定数量的实体
def kg_find_entity():
    file_path = 'kg.json'
    fr = open(file_path, 'r', encoding='utf-8')
    kg_entitys = json.load(fr)
    return kg_entitys


# 最终处理成提交格式，将句子改为和原始数据对应的句子
def generate_submit():
    fr3 = open('val_test.json', 'r', encoding='utf-8')
    sentence = json.load(fr3)
    fr4 = open('submit.json', 'r', encoding='utf-8')
    submit = json.load(fr4)
    fw5 = open('submit_on_system_10_29.json', 'w', encoding='utf-8')
    # {"text": "Molecular diagnostics for sleeping sickness: what is the benefit for the patient?\tSleeping
    #     "entities": [{"entity": "entity", "type": "type", "start": 1, "end": 2},
    #                  {"entity": "entity", "type": "type", "start": 1, "end": 2}]}

    # for tuple in submit:
    #     for sent in sentence:
    #         if sent['context'].replace('\t', '.') == tuple['text']:
    #             f.write(str({"text": sent['context'], "entities": tuple['entities']}) + '\n')
    #             del (sentence[0])
    #             break

    kg_entitys = kg_find_entity()
    # sentence = sentence[:20]
    for sent in sentence:
        dict = {"text": sent['context'], "entities": []}
        for tuple in submit:
            if sent['context'].replace('\t', '.') == tuple['text']:
                dict["entities"] += tuple['entities']
                # if len(tuple['entities']) is not 0:
                submit.remove(tuple)
                break
        context_list = list(sent['context'])
        for entity_str in kg_entitys:
            entity = entity_str[0]
            tag = entity_str[1]
            if tag in ['Drug', 'Phenotype']:
                continue
            if entity in sent['context']:
                for i in re.finditer(entity, sent['context']):
                    start = i.start()
                    end = i.end()
                    # {"entity": "tetanus", "type": "Disease", "start": 14, "end": 21}
                    if start - 1 > 0 and end < len(context_list):
                        if sent['context'][start - 1] is ' ' and sent['context'][end] is ' ':
                            if {"entity": sent['context'][start: end], "type": tag, "start": start, "end": end} not in dict['entities']:
                                dict['entities'].append({"entity": sent['context'][start: end], "type": tag, "start": start, "end": end})
                    elif start == 0 and end < len(sent['context']):
                        if sent['context'][end] is ' ':
                            if {"entity": sent['context'][start: end], "type": tag, "start": start, "end": end} not in dict['entities']:
                                dict['entities'].append({"entity": sent['context'][start: end], "type": tag, "start": start, "end": end})
                    elif start - 1 > 0 and end == len(sent['context']):
                        if sent['context'][start - 1] is ' ':
                            if {"entity": sent['context'][start: end], "type": tag, "start": start, "end": end} not in dict['entities']:
                                dict['entities'].append({"entity": sent['context'][start: end], "type": tag, "start": start, "end": end})
                    else:
                        continue
                # pattern = re.compile("\s{1}\(?" + entity + '\)?\s{1}')
                # for i in re.finditer(pattern, sent['context']):
                #     start = i.start()
                #     end = i.end()
                #     start_pos = start
                #     end_pos = end
                #     # 对正则表达式匹配出的位置进行修改
                #     if sent['context'][start] in [' ', '(', '\t']:
                #         start_pos = start + 1
                #     if sent['context'][start + 1] in [' ', '(', '\t']:
                #         start_pos = start + 2
                #     if sent['context'][end - 1] in [' ', '(', '\t']:
                #         end_pos = end - 1
                #     if sent['context'][end - 2] in [' ', '(', '\t']:
                #         end_pos = end - 2
                #
                #     if {"entity": sent['context'][start_pos: end_pos], "type": tag, "start": start_pos, "end": end_pos} not in dict['entities']:
                #         dict['entities'].append({"entity": sent['context'][start_pos: end_pos], "type": tag, "start": start_pos, "end": end_pos})
        data = json.dumps(dict, ensure_ascii=False)
        if len(dict['entities']) is not 0:
            fw5.write(data + '\n')

    fw5.close()
    print('保存成功')


def caculate_query_len():
    fr = open('../query/en_ccks_covid19.json', 'r', encoding='utf-8')

    type_dict = json.load(fr)

    type_dict = type_dict['default']
    type_len = {}
    for key, value in type_dict.items():
        type_len[key] = len(tokenizer.tokenize(value))

    print(type_len)


type_len = {'Disease': 19, 'Phenotype': 18, 'Drug': 28, 'ChemicalCompound': 9, 'Gene': 11, 'Virus': 23, 'Organization': 14, 'Chemical': 8}


if __name__ == '__main__':
    # caculate_vocab_num()
    # caculate_sent_num()
    # caculate_sent_num_for_bert()
    generate_process_data()
    # generate_result()
    # generate_submit()
    # caculate_query_len()

    # 知识图谱补充
    # generate_kg()






