from __future__ import division
from bs4 import BeautifulSoup
from state_bill_extractors import bill_text_extractor
import os
import codecs
import argparse
import re
import base64
import json
from tika import parser as tp
import traceback
import logging 
from config import DATA_PATH

try:
    from os import scandir, walk
except ImportError:
    from scandir import scandir, walk



def get_first_and_last_bill_documents(json_obj):
    state_code = json_obj['state']

    bill_documents = []
    for v in range(2):

        try:
            bill_document = base64.b64decode(json_obj['versions'][v]['bill_document'])
        except:
            bill_documents.append(None)
            continue

        try:
            mimetype = json_obj['versions'][v]['mimetype']

        except KeyError:
            mimetype = json_obj['versions'][v]['+mimetype']

        url = json_obj['versions'][v]['url']
        # try to extract text with bill-specific extractor
        bill_text = bill_text_extractor(state_code, bill_document, mimetype, url)

        # if fails then try tika extractor as backup
        if not bill_text or len(bill_text) < 1000:

            try:
                bill_text = tp.from_buffer(bill_document)['content']
                #if extraction results in short text, most likely a fail
                if len(bill_text) < 1000:
                    bill_text = None
            except Exception:
                bill_text = None
        
        
        bill_documents.append(bill_text)

    return bill_documents



# extracts text from bill documents fetched from sunlight
# and constructs new json obj with selected meta-data
def extract_bill_document(bill_file_path):
    try:

        bill_dict = {}
        data_dict = json.loads(open(bill_file_path).read())
        
        #test whether a document is a bill or resolution
        bill_text_count = [1 for x in data_dict['type'] if "bill" in x.lower()]
        good_bill_prefixes = ["A","AJ", "AJR","CACR","HB","S","HJR","ACA","HF","SF","HJ","SJ"
                "HJRCA","SJRCA","HSB","IP","LB","SB","SCA","SP"]
        if sum(bill_text_count) < 1 and data_dict['bill_id'].split()[0] not in good_bill_prefixes:
            return
        
        


        # extract first and last versions of bill document
        # and add to json dict
        bill_document_first, bill_document_last = get_first_and_last_bill_documents(data_dict)
        bill_dict['bill_document_first'] = bill_document_first
        bill_dict['bill_document_last'] = bill_document_last
        
        if bill_document_first == None or bill_document_last == None:
            logging.warning("failed to extract text for {0}".format(bill_file_path))
    
        else:
            logging.info("successfully extracted text for {0}".format(bill_file_path))

        # assign attributes that will be used 
        bill_id = re.sub("\s+", "", data_dict['bill_id'])
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
        bill_dict['actions'] = data_dict['actions']
        bill_dict['action_dates'] = data_dict['action_dates']
        bill_dict['date_introduced'] = data_dict['action_dates']['first']
        bill_dict['date_signed'] = data_dict['action_dates']['signed']


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

        return bill_dict
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        trace_message = re.sub("\n+", "\t", traceback.format_exc())
        trace_message = re.sub("\s+", " ", trace_message)
        trace_message = "<<{0}>>".format(trace_message)
        m = "Failed to extract document for {0}: {1}".format(bill_file_path, trace_message)
        logging.error(m)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('command', help='command to run, options are: build_index')
    parser.add_argument('--data_path', dest='data_path', help="file path of data to be indexed ")

    args = parser.parse_args()

    #extracts text from bill documents and populates a json file with a json_object per row
    if args.command == "extract_bills":
        #configure logging 
        logging.getLogger('tp').setLevel(logging.ERROR)
        logging.getLogger('requests').setLevel(logging.ERROR)
        logging.basicConfig(filename=os.environ['POLICY_DIFFUSION'] + '/logs/state_bill_extractor.log',
                level=logging.DEBUG)
        
        bill_files = []
        for dirname, dirnames, filenames in walk(args.data_path):
            for filename in filenames:
                bill_files.append(os.path.join(dirname, filename))

        outFile = codecs.open("{0}/extracted_data/extracted_bills.json".format(DATA_PATH), 'w')
        for i, bill_file in enumerate(bill_files):
            bill_json_obj = extract_bill_document(bill_file)

            outFile.write("{0}\n".format(json.dumps(bill_json_obj)))

        outFile.close()



##extracts text from model legislation
def extract_model_legislation(json_file, encoded):
    '''
    Keyword Args: 
    json_file: corresponds to json file with model legislation
    encoded: True/False if json file is b64 encoded

    returns:
        dictionary with url, date, and text of model legislation 
    decription:
        extract text from model legislation  
    '''
    data = []
    with open(json_file) as f:
        for line in f:
            data.append(json.loads(line))

    model_legislation = {}
    for i in range(len(data)):
        model_legislation[i] = data[i]

    if encoded == True:
            for i in range(len(model_legislation)):
                try:
                    ml = model_legislation[i]['source']
                    ml = base64.b64decode(ml)
                    ml = tp.from_buffer(ml)
                    model_legislation[i]['source'] = ml['content']
                except AttributeError:
                    model_legislation[i]['source'] = None
            return model_legislation
            
    else:
        return model_legislation

    
 
