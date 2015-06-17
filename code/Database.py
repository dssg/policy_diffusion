# This module handles the creation and management of the elasticsearch backend


import os
import argparse
from elasticsearch import Elasticsearch
import ujson
import re
from tika import parser
import base64
import logging

from pprint import pprint

#DATABASE_LOG = os.environ['POLICY_DIFFUSION'] + '/logs/database.log'
#logging.basicConfig(filename = DATABASE_LOG, level=logging.DEBUG)

STATE_BILL_INDEX = "state_bills"
MODEL_LEGISLATION_INDEX = "model_legistlation"

ES_CONNECTION = Elasticsearch()

# bulk loads all json files in subdirectory
def load_bulk_bills(bill_directory):
    bill_files = []
    for dirname, dirnames, filenames in os.walk(bill_directory):
        for filename in filenames:
            bill_files.append(os.path.join(dirname, filename))

    bulk_data = []
    for i, bill_file in enumerate(bill_files):

        data_dict = ujson.decode(open(bill_file).read())
        bill_dict = {}

        bill_text_count = [1 for x in data_dict['type'] if "bill" in x.lower()]
        if sum(bill_text_count) < 1:
            continue

        bill_id = re.sub("\s+", "", data_dict['bill_id'])

        try:
            if data_dict['versions'] == []:
                bill_document_first = ""
                bill_document_last = ""
            else:
                bill_document_first = base64.b64decode(data_dict['versions'][0]['bill_document'])
                bill_document_first = parser.from_buffer(bill_document_first)['content']
                bill_document_last = base64.b64decode(data_dict['versions'][-1]['bill_document'])
                bill_document_last = parser.from_buffer(bill_document_last)['content']

        except Exception as e:
            print e.args
            bill_document_first = None
            bill_document_last = None


        bill_dict['unique_id'] = "{0}_{1}_{2}".format(data_dict['state'], data_dict['session'], bill_id)
        bill_dict['bill_id'] = data_dict['bill_id']
        bill_dict['date_updated'] = data_dict['updated_at']
        bill_dict['session'] = data_dict['session']
        bill_dict['sunlight_id'] = data_dict['id']
        bill_dict['bill_title'] = data_dict['title']
        bill_dict['bill_type'] = data_dict['type']
        bill_dict['state'] = data_dict['state']
        bill_dict['chamber'] = data_dict['chamber']
        bill_dict['date_created'] = data_dict['created_at']
        bill_dict['bill_document_first'] = bill_document_first
        bill_dict['bill_document_last'] = bill_document_last
        bill_dict['actions'] = data_dict['actions']


        if "short_tite" in data_dict.keys():
            bill_dict['short_title'] = data_dict['short_title']
        elif "+short_title" in data_dict.keys():
            bill_dict['short_title'] = data_dict['+short_title']

        else:
            bill_dict['short_title'] = None

        if "summary" in data_dict.keys():
            bill_dict['summary'] = data_dict['summary']
        else:
            bill_dict['summary'] = None

        op_dict = {
            "index": {
                "_index": STATE_BILL_INDEX,
                "_type": "bill_document",
                "_id": bill_dict["unique_id"]
            }
        }

        bulk_data.append(op_dict)
        bulk_data.append(bill_dict)


        if (i%100) == 0:
            ES_CONNECTION.bulk(index=STATE_BILL_INDEX, body=bulk_data,timeout = 100)
            bulk_data = []


    ES_CONNECTION.bulk(index=STATE_BILL_INDEX, body=bulk_data,timeout=100)
    return


# creates index for bills and model legislation stored in
# data_path, overwriting index if it is already created
def create_index(data_path):
    bill_data_path = os.path.join(data_path, "scraped_bills")
    state_directories = [os.path.join(bill_data_path, x) for x in os.listdir(bill_data_path)]

    if ES_CONNECTION.indices.exists(STATE_BILL_INDEX):
        print("deleting '%s' index..." % (STATE_BILL_INDEX))
        res = ES_CONNECTION.indices.delete(index=STATE_BILL_INDEX)

    request_body = {"settings": {"number_of_shards": 1, "number_of_replicas": 0}}

    print("creating '%s' index..." % (STATE_BILL_INDEX))
    res = ES_CONNECTION.indices.create(index=STATE_BILL_INDEX, body=
    {"settings": {"number_of_shards": 1, "number_of_replicas": 0}})

    for state_dir in state_directories:
        print ("Loading in {0}".format(state_dir))
        load_bulk_bills(state_dir)


## main functiont that manages unix interface
def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('command', help='command to run, options are: build_index')
    parser.add_argument('--data_path', dest='data_path', help="file path of data to be indexed ")

    args = parser.parse_args()

    if args.command == "build_index":
        create_index(args.data_path)
    else:
        print args


if __name__ == "__main__":
    main()
