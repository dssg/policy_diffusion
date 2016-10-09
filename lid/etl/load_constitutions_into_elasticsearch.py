#!/bin/python

import time
import glob
import json
import requests
from io import open
from elasticsearch import Elasticsearch

files = glob.glob("*.txt")
es = Elasticsearch([{'host': "54.244.236.175", 'port': 9200}])

for file in files:
    print file
    state_year = file.split(".")[0]
    state = state_year[:-5]
    year = int(state_year[-4:])
    file_text = open(file, 'r', encoding='ISO-8859-1').read()
    json_object = {
            "document_type": "constitution",
            "state": state,
            "year": year,
            "constitution": file_text
    }

    es.index(index="constitutions", doc_type="constitution", id=state_year, body=json.dumps(json_object))
    time.sleep(1)
