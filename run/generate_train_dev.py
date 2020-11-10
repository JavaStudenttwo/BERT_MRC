#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from random import shuffle

q_repo = "query"
path_en_ccks_covid19 = os.path.join(q_repo, "en_ccks_covid19.json")

def load_query_map(query_map_path):
    with open(query_map_path, "r", encoding='utf-8') as f:
        query_map = json.load(f)
    return query_map

query_en_ccks_covid19 = load_query_map(path_en_ccks_covid19)

queries_for_dataset = {
    "en_ccks_covid19": query_en_ccks_covid19
}


def generate_flat_ner_dta():

    dataset_name = "en_ccks_covid19"
    query_sign = "default"

    source_file_path = os.path.join("origin_data", "train.json")
    # source_data = load_conll(train, source_file_path)
    with open(source_file_path, "r", encoding='utf-8') as f:
        source_data = json.load(f)

    shuffle(source_data)
    sent_len = int((len(source_data) / 5) * 4)
    train_source_data = source_data[:sent_len]
    dev_source_data = source_data[sent_len:]

    target_file_path = os.path.join("data", "mrc-ner.train")
    generate_query_ner_dataset_(True, target_file_path, train_source_data, dataset_name=dataset_name,
                               query_sign=query_sign)

    target_file_path = os.path.join("data", "mrc-ner.dev")
    generate_query_ner_dataset_(True, target_file_path, dev_source_data, dataset_name=dataset_name,
                               query_sign=query_sign)


def generate_query_ner_dataset_(train, dump_file_path, source_data,
    dataset_name=None, query_sign="default"):
    """
    Args:
        source_data_file: /data/genia/train.word.json | /data/msra/train.char.bmes
        dump_data_file: /data/genia-mrc/train.mrc.json | /data/msra-mrc/train.mrc.json
        dataset_name: one in ["en_ontonotes5", "en_conll03", ]
        entity_sign: one of ["nested", "flat"]
        query_sign: defualt is "default"
    Desc:
        pass
    """
    entity_queries = queries_for_dataset[dataset_name][query_sign]
    label_lst = queries_for_dataset[dataset_name]["labels"]

    target_data = transform_examples_to_qa_features(entity_queries, label_lst, source_data)

    with open(dump_file_path, "w", encoding='utf-8') as f:
        json.dump(target_data, f, sort_keys=True, ensure_ascii=False, indent=2)


def transform_examples_to_qa_features(query_map, entity_labels, data_instances):
    """
    Desc:
        convert_examples to qa features
    Args:
        query_map: {entity label: entity query};
        data_instance
    """
    mrc_ner_dataset = []
    tmp_qas_id = 0
    for idx, data_item in enumerate(data_instances):
        tmp_query_id = 0
        for label_idx, tmp_label in enumerate(entity_labels):
            tmp_query_id += 1
            tmp_query = query_map[tmp_label]
            tmp_context = data_item["context"]

            tmp_start_pos = []
            tmp_end_pos = []
            tmp_entity_pos = []

            start_end_label = data_item["label"][tmp_label] if tmp_label in data_item["label"].keys() else -1

            if start_end_label == -1:
                tmp_impossible = True
            else:
                for start_end_item in data_item["label"][tmp_label]:
                    start_end_item = start_end_item.replace(",", ";")
                    start_idx, end_idx = [int(ix) for ix in start_end_item.split(";")]
                    tmp_start_pos.append(start_idx)
                    tmp_end_pos.append(end_idx)
                    tmp_entity_pos.append(start_end_item)
                tmp_impossible = False

            mrc_ner_dataset.append({
                "qas_id": "{}.{}".format(str(tmp_qas_id), str(tmp_query_id)),
                "query": tmp_query,
                "context": tmp_context,
                "entity_label": tmp_label,
                "start_position": tmp_start_pos,
                "end_position": tmp_end_pos,
                "span_position": tmp_entity_pos,
                "impossible": tmp_impossible
            })
        tmp_qas_id += 1

    return mrc_ner_dataset





