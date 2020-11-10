#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

import os
import json

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
