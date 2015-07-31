import json
import codecs
import base64
import logging
import re
import os
import sys
import multiprocessing
import utils
import random
import argparse
import traceback
import urllib2
from config import DATA_PATH
from bs4 import BeautifulSoup
import numpy as np

try:
    from os import scandir, walk
except ImportError:
    from scandir import scandir, walk


state_code= sys.argv[1]
if state_code == "all_states":
    bill_data_path = "{0}/dssg/scraped_bills_new".format(DATA_PATH)
else:
    bill_data_path = "{0}/dssg/scraped_bills_new/{1}".format(DATA_PATH,state_code)


bill_file_paths = []
for dirname, dirnames, filenames in walk(bill_data_path):
    for filename in filenames:
        bill_file_paths.append(os.path.join(dirname, filename))

print len(bill_file_paths)
raw_input("ready??")
no_docs = 0
bad_docs = 0
doc_lengths = []
bad_link_doc = open("/home/mburgess/bad_links_for_paul.txt",'w')
for i,path in enumerate( bill_file_paths ):

    json_obj = json.load(open(path))

    doc_1 = json_obj['versions'][0]['bill_document']
    doc_2 = json_obj['versions'][-1]['bill_document']

    if doc_1 is None:
        no_docs += 1
    if doc_2 is None:
        no_docs += 1
        bad_link_doc.write("http://static.openstates.org/documents/{0}/{1}\n".format(json_obj['state'],json_obj['versions'][-1]['doc_id']))

    if doc_1 is None or doc_2 is None:
        continue

    doc_1 = base64.b64decode(doc_1)
    doc_2 = base64.b64decode(doc_2)

    if len(doc_1) < 1000:
        no_docs +=1
    if len(doc_2) < 1000:
        no_docs +=1

    doc_lengths.append(len(doc_1))
    doc_lengths.append(len(doc_2))



print "num no_docs",no_docs
print "min length",min(doc_lengths)
print "avg length",np.mean(doc_lengths)
print "max length",np.max(doc_lengths)