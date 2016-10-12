#!/usr/bin/env python

# Written for Python 2.7

from lid import LID
from text_alignment import AffineLocalAligner,LocalAligner
import database
import json
import base64
import codecs
import re
import logging
import os
import traceback
import sys
from database import ElasticConnection
from elasticsearch import Elasticsearch
import time

def get_constitution_alignments(query_doc):
    result_docs = constitution_lidy.find_constitution_alignments(
            query_doc,
            document_type = "text",
            split_sections = True,
            query_document_id = "text")
    return result_docs
 

def main():

    docs = ec.get_all_doc_ids('constitutions')

    for doc in docs:
        print doc
        doc_text = es_connection.get_source(index = 'constitutions', id = doc)['constitution']
        result_doc = get_constitution_alignments(doc_text)
        open('/mnt/data/jwalsh/constitution_matches.json', 'a').write(json.dumps(result_doc))
        time.sleep(1)



if __name__ == "__main__":
    #elastic host ip
    ip_addy = os.environ['ELASTICSEARCH_IP']

    #instantiate lid,aligner and elasticsearch objects
    aligner = AffineLocalAligner(match_score=4, mismatch_score=-1, gap_start=-3, gap_extend = -1.5)
    ec = ElasticConnection(host = ip_addy)
    es_connection = Elasticsearch([{'host': ip_addy, 'port': 9200}])

    query_results_limit = os.environ['QUERY_RESULTS_LIMIT']
    constitution_lidy = LID(query_results_limit=query_results_limit, elastic_host=ip_addy,
	lucene_score_threshold=0.01, aligner=aligner)

    main()

