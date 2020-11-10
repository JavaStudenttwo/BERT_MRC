#!/usr/bin/env python3
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# description:
# evaluate 


import random 
import argparse 
import numpy as np 
import json

import torch 
from data_loader.model_config import Config 
from data_loader.mrc_data_processor import Conll03Processor, MSRAProcessor, Onto4ZhProcessor, Onto5EngProcessor, GeniaProcessor, ACE2004Processor, ACE2005Processor, ResumeZhProcessor, CCKSProcessor, CCKSTask2Processor, KGCovid19Processor
from data_loader.mrc_data_loader import MRCNERDataLoader
from model.bert_mrc import BertQueryNER 
from data_loader.bert_tokenizer import BertTokenizer4Tagger 
from metric.mrc_ner_evaluate  import flat_ner_performance, nested_ner_performance



def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", default="../config/en_bert_base_uncased.json", type=str)
    parser.add_argument("--data_dir", default='../data_dir/KG_Covid19_Task1/data', type=str)
    parser.add_argument("--bert_model", default="../../../../bert_model/bert-base-uncased", type=str, )
    parser.add_argument("--saved_model", type=str, default="../output/llGJgz.bin")
    parser.add_argument("--max_seq_length", default=400, type=int)
    parser.add_argument("--test_batch_size", default=4, type=int)
    parser.add_argument("--data_sign", type=str, default="KGCovid19Processor")
    parser.add_argument("--entity_sign", type=str, default="flat")
    parser.add_argument("--n_gpu", type=int, default=1) 
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--weight_start", type=float, default=1.0) 
    parser.add_argument("--weight_end", type=float, default=1.0)
    parser.add_argument("--weight_span", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--entity_threshold", type=float, default=0.5)
    parser.add_argument("--data_cache", type=bool, default=False)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.cuda.manual_seed_all(args.seed)

    return args 


def load_data(config):

    print("-*-"*10)
    print("current data_sign: {}".format(config.data_sign))

    if config.data_sign == "conll03":
        data_processor = Conll03Processor() 
    elif config.data_sign == "zh_msra":
        data_processor = MSRAProcessor()
    elif config.data_sign == "zh_onto":
        data_processor = Onto4ZhProcessor()
    elif config.data_sign == "en_onto":
        data_processor = Onto5EngProcessor()
    elif config.data_sign == "genia":
        data_processor = GeniaProcessor()
    elif config.data_sign == "ace2004":
        data_processor = ACE2004Processor()
    elif config.data_sign == "ace2005":
        data_processor = ACE2005Processor()
    elif config.data_sign == "zh_ccks_task01":
        data_processor = CCKSProcessor()
    elif config.data_sign == "zh_ccks_task02":
        data_processor = CCKSTask2Processor()
    elif config.data_sign == "KGCovid19Processor":
        data_processor = KGCovid19Processor()
    elif config.data_sign == "resume":
            data_processor = ResumeZhProcessor()
    else:
        raise ValueError("Please Notice that your data_sign DO NOT exits !!!!!") 


    label_list = data_processor.get_labels()
    tokenizer = BertTokenizer4Tagger.from_pretrained(config.bert_model, do_lower_case=True)


    dataset_loaders = MRCNERDataLoader(config, data_processor, label_list, tokenizer, mode="test", allow_impossible=True)

    test_dataloader = dataset_loaders.get_dataloader(data_sign="test")

    return test_dataloader, label_list 



def merge_config(args_config):
    model_config_path = args_config.config_path 
    model_config = Config.from_json_file(model_config_path)
    model_config.update_args(args_config)
    model_config.print_config()

    return model_config 


def main():
    args_config = args_parser()
    config = merge_config(args_config)
    test_loader, label_list = load_data(config)
    model, device, n_gpu = load_model(config, label_list,)

    acc, pre, rec, f1 = eval_checkpoint(model, test_loader, config, device, n_gpu, label_list, eval_sign="test")
    


def load_model(config, label_list):
    device = torch.device("cuda")
    n_gpu = config.n_gpu 
    model = BertQueryNER(config)
    checkpoint = torch.load(config.saved_model)
    model.load_state_dict(checkpoint)
    model.to(device)
    if config.n_gpu >1 :
        model = torch.nn.DataParallel(model)

    return model, device, n_gpu 



def eval_checkpoint(model_object, eval_dataloader, config, \
    device, n_gpu, label_list, eval_sign="test"):
    
    model_object.eval()

    start_pred_lst = []
    end_pred_lst = []
    span_pred_lst = []
    mask_lst = []

    start_gold_lst = []
    end_gold_lst = []
    span_gold_lst = []

    ner_cate_lst = []


    for input_ids, input_mask, segment_ids, start_pos, end_pos, ner_cate in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        start_pos = start_pos.to(device)
        end_pos = end_pos.to(device)
        # span_pos = span_pos.to(device)

        with torch.no_grad():
            start_logits, end_logits = model_object(input_ids, segment_ids, input_mask)

        start_pos = start_pos.to("cpu").numpy().tolist()
        end_pos = end_pos.to("cpu").numpy().tolist()
        # span_pos = span_pos.to("cpu").numpy().tolist()

        start_label = start_logits.detach().cpu().numpy().tolist()
        end_label = end_logits.detach().cpu().numpy().tolist()
        # span_label = span_logits.detach().cpu().numpy().tolist()

        input_mask = input_mask.to("cpu").detach().numpy().tolist()

        ner_cate_lst += ner_cate.numpy().tolist()
        mask_lst += input_mask 

        start_pred_lst += start_label 
        end_pred_lst += end_label 
        # span_pred_lst += span_label
        
        start_gold_lst += start_pos 
        end_gold_lst += end_pos 
        # span_gold_lst += span_pos

    span_pred_lst = [[[1] * len(start_gold_lst[0])] * len(start_gold_lst[0])] * len(start_gold_lst)
    span_gold_lst = [[[1] * len(start_gold_lst[0])] * len(start_gold_lst[0])] * len(start_gold_lst)

    if config.entity_sign == "flat":
        acc, pre, rec, f1, pred_span_triple_lst = flat_ner_performance(start_pred_lst, end_pred_lst, span_pred_lst, start_gold_lst, end_gold_lst, span_gold_lst, ner_cate_lst, label_list, threshold=config.entity_threshold, dims=2)
    else:
        acc, pre, rec, f1, pred_span_triple_lst = nested_ner_performance(start_pred_lst, end_pred_lst, span_pred_lst, start_gold_lst, end_gold_lst, span_gold_lst, ner_cate_lst, label_list, threshold=config.entity_threshold, dims=2)

    pred_entity_list = []
    # 根据预测的序列标签得到
    if config.entity_sign == "flat":
        for tags in pred_span_triple_lst:
            entity_list = []
            entity = {}
            index = 0
            while index < len(tags):
                tag = tags[index]
                if tag != 'O':
                    entity['begin'] = index
                    start_pos, type = tag.split('-')
                    while tag != 'O':
                        index += 1
                        tag = tags[index]
                    end_pos, type = tags[index - 1].split('-')
                    if start_pos == 'B' and end_pos == 'E':
                        entity['end'] = index
                        entity['tag'] = type
                        entity_list.append(entity)
                        entity = {}
                index += 1
            pred_entity_list.append(entity_list)

    # 组织答案
    span_triple_lst = [[] for i in range(len(pred_span_triple_lst))]
    for i, j in zip(pred_entity_list, span_triple_lst):
        for z in i:
            j.append({'tag': z['tag'], 'begin': z['begin'], 'end': z['end']})

    with open('result.json', 'w', encoding='utf-8') as file_obj:
        json.dump(span_triple_lst, file_obj, ensure_ascii=False)

    print('保存成功')
    print("=*="*10)
    print("eval: acc, pre, rec, f1")
    print(acc, pre, rec, f1)

    return  acc, pre, rec, f1 




if __name__ == "__main__":
    main()





