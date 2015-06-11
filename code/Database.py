# This module handles the creation and management of the elasticsearch backend


import os
import argparse
from elasticsearch import Elasticsearch
import ujson
import re
from tika import parser
import base64
from pprint import pprint

STATE_BILL_INDEX = "state_bills"
MODEL_LEGISLATION_INDEX = "model_legistlation"


# creates index for bills and model legislation stored in
# data_path, overwriting index if it is already created
def create_index(data_path):

    #get file location of all json files
    #bill_data_path = os.path.join(data_path, "scraped_bills")
    bill_data_path = os.path.join(data_path, "scraped_bills/ca")
    bill_files = []
    for dirname, dirnames, filenames in os.walk(bill_data_path):
        for filename in filenames:
            bill_files.append( os.path.join(dirname, filename) )

    bulk_data = []
    for i,bill_file in enumerate(bill_files[8500:]):
        print i

        data_dict = ujson.decode(open(bill_file).read())
        bill_dict = {}

        print "______________________________________________________________________"
        pprint( data_dict.keys())
        print bill_file
        print "______________________________________________________________________"

        bill_text_count = [1 for x in data_dict['type'] if "bill" in x.lower()]
        if sum(bill_text_count) < 1:
            continue

        bill_id = re.sub("\s+","",data_dict['bill_id'])

        bill_document_first = base64.b64decode(data_dict['versions'][0]['bill_document'])
        bill_document_last = base64.b64decode(data_dict['versions'][-1]['bill_document'])

        if len(bill_document_first) == 0:
            bill_document_first = None
        else:
            bill_document_first = parser.from_buffer(bill_document_first)['content']

        if len(bill_document_last) == 0:
            bill_document_last = None
        else:
            bill_document_last = parser.from_buffer(bill_document_last)['content']



        bill_dict['unique_id'] = "{0}_{1}_{2}".format(data_dict['state'],data_dict['session'],bill_id)
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


    es = Elasticsearch()

    if es.indices.exists(STATE_BILL_INDEX):
        print("deleting '%s' index..." % (STATE_BILL_INDEX))
        res = es.indices.delete(index = STATE_BILL_INDEX)
        print(" response: '%s'" % (res))

    res = es.bulk(index = STATE_BILL_INDEX, body = bulk_data, refresh = True)



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
