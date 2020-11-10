#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

import os
import sys


from data_dir.KG_Covid19_Task1.generate_mrc_dataset import generate_query_ner_dataset
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# 加载词典 pre-trained model tokenizer (vocabulary)
VOCAB = './bert-large-cased-vocab.txt'
tokenizer = BertTokenizer.from_pretrained(VOCAB)


def test_flat_ner():
    entity_sign = "flat"
    dataset_name = "en_ccks_covid19"
    query_sign = "default"

    source_file_path = os.path.join("process", "train.json")
    target_file_path = os.path.join("data", "mrc-ner.train")
    generate_query_ner_dataset(True, tokenizer, source_file_path, target_file_path, entity_sign=entity_sign, dataset_name=dataset_name,
                               query_sign=query_sign)

    source_file_path = os.path.join("process", "val_test.json")
    target_file_path = os.path.join("data", "mrc-ner.dev")
    generate_query_ner_dataset(True, tokenizer, source_file_path, target_file_path, entity_sign=entity_sign,
                               dataset_name=dataset_name,
                               query_sign=query_sign)


if __name__ == "__main__":
    # test_nested_ner()
    test_flat_ner()