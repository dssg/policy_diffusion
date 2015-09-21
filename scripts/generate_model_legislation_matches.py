#!/opt/anaconda/bin/python

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
from utils.general_utils import deadline,TimedOutExc
import time



@deadline(1000)
def get_alignments(model_doc):
    result_docs = lidy.find_state_bill_alignments(model_doc['source'],document_type = "model_legislation",
            split_sections = True,query_document_id = model_doc['id'])
    return result_docs


def test(model_doc):
    return model_doc


if __name__ == "__main__":
    
    #elastic host ip
    ip_addy = "54.203.12.145"



    #configure logging
    logging.basicConfig(filename="{0}/logs/model_legislation_alignment.log".format(os.environ['POLICY_DIFFUSION']),
                level=logging.DEBUG)
    logging.getLogger('elasticsearch').setLevel(logging.ERROR)
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    logging.getLogger('json').setLevel(logging.ERROR)
    
    
    #instantiate lid object
    
    aligner = AffineLocalAligner(match_score=4, mismatch_score=-1, gap_start=-3, gap_extend = -1.5)

    lidy = LID(query_results_limit=100,elastic_host = ip_addy,lucene_score_threshold = 0.1,aligner = aligner)
    
    for line in sys.stdin:
        model_doc = json.loads(line.strip())
        
        try:
            result_doc = get_alignments(model_doc)
            #result_doc = test(model_doc)
            print json.dumps(result_doc)
        
        except (KeyboardInterrupt, SystemExit):
            raise
        except TimedOutExc:
            m = "timeout error query_id {0}: {1}".format(model_doc['id'], trace_message)
            logging.error(m)
            print json.dumps({"query_document_id": model_doc['id'],"error":"timeout error"})

        except:
            trace_message = re.sub("\n+", "\t", traceback.format_exc())
            trace_message = re.sub("\s+", " ", trace_message)
            trace_message = "<<{0}>>".format(trace_message)
            m = "random error query_id {0}: {1}".format(model_doc['id'], trace_message)
            logging.error(m)
            print json.dumps({"query_document_id": model_doc['id'],"error":"trace_message"})

