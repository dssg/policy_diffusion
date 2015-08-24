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
from database import ElasticConnection
import time

class NoneDocException(Exception):
    pass


@deadline(1000)
def get_alignments(query_doc,bill_id):
    result_docs = lidy.find_state_bill_alignments(query_doc,document_type = "state_bill",
            split_sections = True,state_id = bill_id[0:2],query_document_id = bill_id)
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
    
    
    #instantiate lid,aligner and elasticsearch objects
    
    aligner = AffineLocalAligner(match_score=4, mismatch_score=-1, gap_start=-3, gap_extend = -1.5)
    
    ec = ElasticConnection(host = ip_addy)

    lidy = LID(query_results_limit=100,elastic_host = ip_addy,lucene_score_threshold = 0.1,aligner = aligner)
    
    #for line in sys.stdin:
    
    try:
        
        bill_id = sys.argv[1]
        query_doc =  ec.get_bill_by_id(bill_id)['bill_document_last']
        
        if query_doc is None:
            raise NoneDocException
        
        result_doc = get_alignments(query_doc,bill_id)
        logging.info("obtained alignments for {0}".format(bill_id))
        print json.dumps(result_doc)
    
    except (KeyboardInterrupt, SystemExit):
        raise
    
    except NoneDocException:
        
        m = "none doc error query_id {0}: {1}".format(bill_id, "None doc error")
        logging.error(m)
        print json.dumps({"query_document_id": bill_id,"error":"none doc error"})

    except TimedOutExc:
        
        m = "timeout error query_id {0}: {1}".format(bill_id, "timeout error")
        logging.error(m)
        print json.dumps({"query_document_id": bill_id,"error":"timeout error"})

    except:

        trace_message = re.sub("\n+", "\t", traceback.format_exc())
        trace_message = re.sub("\s+", " ", trace_message)
        trace_message = "<<{0}>>".format(trace_message)
        m = "random error query_id {0}: {1}".format(bill_id, trace_message)
        logging.error(m)
        print json.dumps({"query_document_id": bill_id,"error":"trace_message"})
