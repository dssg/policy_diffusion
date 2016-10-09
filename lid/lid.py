from __future__ import division


'''main module for the LID (legislative influence detector) system '''
from database import ElasticConnection
from multiprocessing import Pool
from text_alignment import LocalAligner
from utils.text_cleaning import clean_document
from utils.general_utils import alignment_tokenizer
import argparse
import json
import logging
import os
import re
import time
import traceback


#configure logging options here

logging.basicConfig(filename="{0}/lid.log".format(os.environ['LOGFILE_DIRECTORY']),
                level=logging.DEBUG)
logging.getLogger('elasticsearch').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)


'''custom exception object for LID class'''
class LidException(Exception):
    pass



class LID(object):
    '''LID class that contains all of the central functionality for querying and aligning text between state
    bills and model legistlation'''

    
    def __init__(self,aligner = LocalAligner(),elastic_host = "localhost",elastic_port=9200,
            query_results_limit=300,lucene_score_threshold = 0.01):
        '''
        alignment_object: any object that inherets from abstract class Alignment

        elastic_host: ip adress of the elastic search instance (default: localhost)

        elastic_port: port of the elastic search instance (defualt: 9200)

        num_results_limit: the limit on the number of results to return with the elastic search query (default: 100)
        
        '''
        self.aligner = aligner
        self.elastic_connection = ElasticConnection(host = elastic_host,port = elastic_port)
        self.results_limit = query_results_limit
        self.lucene_score_threshold = lucene_score_threshold
        
    

    def find_state_bill_alignments(self, query_document, document_type="text", split_sections=False, **kwargs):
        '''
        query_document: query document, usually in the form of an entire bill, model legistlation or segment of either
        
        document_type: specifies the document type, default: "text" means that know section chunking will be done
                        on the query, other options include state bill tuples i.e ("state_bill","al")
                        and "model_legislation"

        split_sections: specifies whether the query document will be broken into sections to find multiple alignments
                        (True) or whether to treat the documents as one and identify a single best alignment (False)
                            
        '''

        if document_type == "state_bill":
            try:
                kwargs['state_id']
                kwargs['query_document_id'] 
            except KeyError:
                raise LidException(
                        "if document type is state_bill then you musy specify state_id and query_document_id")

        elif document_type == "model_legislation":
            try:
                kwargs['query_document_id'] 
            except KeyError:
                raise LidException("if document type is model_legislation then you musy specify query_document_id")
        
        elif document_type == "text":
            kwargs['query_document_id'] = None
        
        
        query_document = clean_document(query_document,doc_type = document_type,
                    split_to_section = split_sections, **kwargs)
        
        elastic_query = u" ".join(query_document)

        #query elastic search
        result_docs = self.elastic_connection.similar_doc_query(elastic_query,num_results = self.results_limit,
                return_fields = ["state","bill_document_last"])

        align_doc = [alignment_tokenizer(s) for s in query_document]
        
        alignment_docs = {}
        alignment_docs['query_document'] = query_document
        alignment_docs['query_document_id'] = kwargs['query_document_id']
        alignment_docs['alignment_results'] = []

        num_states = 0
        for i,result_doc in enumerate(result_docs):
            #print i,result_doc['score'],result_doc['state']
            
            if "state_id" in kwargs:
                if result_doc['state'] == kwargs['state_id']:
                    continue

            if result_doc['score'] < self.lucene_score_threshold:
                break
            
            result_sequence = clean_document(result_doc['bill_document_last'],state_id = result_doc['state'])[0]
            result_sequence = alignment_tokenizer(result_sequence)
            
            alignment_obj = self.aligner.align(align_doc,[result_sequence])
            
            alignment_doc = {}
            alignment_doc['alignments'] = [x for x in alignment_obj]
            alignment_doc['lucene_score'] = result_doc['score']
            alignment_doc['document_id'] = result_doc['id']
            alignment_docs['alignment_results'].append(alignment_doc)
        
        return alignment_docs




    def find_model_legislation_alignments(self, query_document, document_type="text", split_sections=False, **kwargs):
        '''
        query_document: query document, usually in the form of an entire bill, model legislation or segment of either
        
        document_type: specifies the document type, default: "text" means that know section chunking will be done
                        on the query, other options include state bill tuples i.e ("state_bill","al")
                        and "model_legislation"

        split_sections: specifies whether the query document will be broken into sections to find multiple alignments
                        (True) or whether to treat the documents as one and identify a single best alignment (False)
        '''

        if document_type == "text":
            kwargs['query_document_id'] = None
        
        
        query_document = clean_document(query_document,doc_type = document_type,
                    split_to_section = split_sections, **kwargs)
        
        elastic_query = u" ".join(query_document)

        #query elastic search
        result_docs = self.elastic_connection.similar_doc_query(elastic_query,num_results = self.results_limit,
                return_fields = ["id","source"], index="model_legislation", 
                fields="source")

        align_doc = [alignment_tokenizer(s) for s in query_document]
        
        alignment_docs = {}
        alignment_docs['query_document'] = query_document
        alignment_docs['query_document_id'] = kwargs['query_document_id']
        alignment_docs['alignment_results'] = []

        num_states = 0
        for i,result_doc in enumerate(result_docs):
            #print i,result_doc['score'],result_doc['state']
            
            if "state_id" in kwargs:
                if result_doc['state'] == kwargs['state_id']:
                    continue

            if result_doc['score'] < self.lucene_score_threshold:
                break
            
            result_sequence = clean_document(result_doc['source'],state_id = result_doc['id'])[0]
            result_sequence = alignment_tokenizer(result_sequence)
            
            alignment_obj = self.aligner.align(align_doc,[result_sequence])
            
            alignment_doc = {}
            alignment_doc['alignments'] = [x for x in alignment_obj]
            alignment_doc['lucene_score'] = result_doc['score']
            alignment_doc['document_id'] = result_doc['id']
            alignment_docs['alignment_results'].append(alignment_doc)
        
        return alignment_docs


    def find_constitution_alignments(self, query_document, document_type="text", split_sections=False, **kwargs):
        '''
        query_document: query document, usually in the form of an entire constitution
        
        document_type: specifies the document type, default: "text" means that know section chunking will be done
                        on the query, other options include constitution tuples i.e ("constitution")

        split_sections: specifies whether the query document will be broken into sections to find multiple alignments
                        (True) or whether to treat the documents as one and identify a single best alignment (False)
        '''

        if document_type == "text":
            kwargs['_id'] = None
        
        
        query_document = clean_document(query_document, doc_type=document_type,
                    split_to_section=split_sections, **kwargs)
        
        elastic_query = u" ".join(query_document)

        #query elastic search
        result_docs = self.elastic_connection.similar_doc_query(elastic_query, num_results=self.results_limit,
                return_fields = ["state","year","constitution"], index="constitutions", 
                fields="constitution")

        align_doc = [alignment_tokenizer(s) for s in query_document]
        
        alignment_docs = {}
        alignment_docs['query_document'] = query_document
        alignment_docs['query_document_id'] = kwargs['query_document_id']
        alignment_docs['alignment_results'] = []

        num_states = 0
        for i,result_doc in enumerate(result_docs):
            #print i,result_doc['score'],result_doc['state']

            if result_doc['score'] < self.lucene_score_threshold:
                break
            
            result_sequence = clean_document(result_doc['constitution'],state_id = result_doc['state'])[0]
            result_sequence = alignment_tokenizer(result_sequence)
            
            alignment_obj = self.aligner.align(align_doc,[result_sequence])
            
            alignment_doc = {}
            alignment_doc['alignments'] = [x for x in alignment_obj]
            alignment_doc['lucene_score'] = result_doc['score']
            alignment_doc['document_id'] = result_doc['id']
            alignment_docs['alignment_results'].append(alignment_doc)
        
        return alignment_docs


    def find_evaluation_alignments(self,query_document,document_type = "text",split_sections = False,**kwargs):
        '''
        query_document: query document, usually in the form of an entire bill, model legistlation or segment of either
        
        document_type: specifies the document type, default: "text" means that know section chunking will be done
                        on the query, other options include state bill tuples i.e ("state_bill","al")
                        and "model_legislation"

        split_sections: specifies whether the query document will be broken into sections to find multiple alignments
                        (True) or whether to treat the documents as one and identify a single best alignment (False)
                            
        '''

        if document_type == "state_bill":
            try:
                kwargs['state_id']
                kwargs['query_document_id'] 
            except KeyError:
                raise LidException(
                        "if document type is state_bill then you musy specify state_id and query_document_id")

        elif document_type == "model_legistlation":
            try:
                kwargs['query_document_id'] 
            except KeyError:
                raise LidException("if document type is model_legistlation then you musy specify query_document_id")
        
        elif document_type == "text":
            kwargs['query_document_id'] = None
        
        
        query_document = clean_document(query_document,doc_type = document_type,
                    split_to_section = split_sections, **kwargs)
        
        elastic_query = u" ".join(query_document)

        #query elastic search
        result_docs = self.elastic_connection.similar_doc_query(elastic_query,num_results = self.results_limit,
                return_fields = ["state","bill_document_last"], index = "evaluation_texts")

        align_doc = [alignment_tokenizer(s) for s in query_document]
        # print 'align_doc: ', align_doc
        
        alignment_docs = {}
        alignment_docs['query_document'] = query_document
        alignment_docs['query_document_id'] = kwargs['query_document_id']
        alignment_docs['alignment_results'] = []

        num_states = 0
        for i,result_doc in enumerate(result_docs):
            
            if result_doc['score'] < self.lucene_score_threshold:
                break
            
            result_sequence = clean_document(result_doc['bill_document_last'],state_id = result_doc['state'])
            # print 'result_sequence: ', result_sequence

            result_sequence = [alignment_tokenizer(s) for s in result_sequence]
            
            alignment_obj = self.aligner.align(align_doc,result_sequence)
            
            alignment_doc = {}
            alignment_doc['alignments'] = [x for x in alignment_obj]
            alignment_doc['lucene_score'] = result_doc['score']
            alignment_doc['document_id'] = result_doc['id']
            alignment_docs['alignment_results'].append(alignment_doc)
        
        return alignment_docs


    def find_evaluation_texts(self,query_document, match_group, document_type = "text",split_sections = False,**kwargs):
        '''
        match_group represents the group of bills that an evaluation bill 
        belongs to (e.g., all the stand your ground bills)

        description: for evaluating lucene score threshold

        query_document: query document, usually in the form of an entire bill, model legistlation or segment of either
        
        document_type: specifies the document type, default: "text" means that know section chunking will be done
                        on the query, other options include state bill tuples i.e ("state_bill","al")
                        and "model_legislation"

        split_sections: specifies whether the query document will be broken into sections to find multiple alignments
                        (True) or whether to treat the documents as one and identify a single best alignment (False)
                            
        '''

        if document_type == "state_bill":
            try:
                kwargs['state_id']
                kwargs['query_document_id'] 
            except KeyError:
                raise LidException(
                        "if document type is state_bill then you musy specify state_id and query_document_id")

        elif document_type == "model_legistlation":
            try:
                kwargs['query_document_id'] 
            except KeyError:
                raise LidException("if document type is model_legistlation then you musy specify query_document_id")
        
        elif document_type == "text":
            kwargs['query_document_id'] = None
        
        query_document = clean_document(query_document,doc_type = document_type,
                    split_to_section = split_sections, **kwargs)
        
        elastic_query = u" ".join(query_document)

        #query elastic search
        result_docs = self.elastic_connection.similar_doc_query_for_testing_lucene(elastic_query,
                                                match_group,
                                                num_results = self.results_limit,
                                                return_fields = ["state","bill_document_last"], 
                                                index = "evaluation_bills_all_bills")

        # result_docs = self.elastic_connection.similar_doc_query(elastic_query,
        #                                         num_results = self.results_limit,
        #                                         return_fields = ["state","bill_document_last"], 
        #                                         index = "evaluation_bills_all_bills")


        results = []
        for i,result_doc in enumerate(result_docs):

            if result_doc['score'] < self.lucene_score_threshold:
                break

            results.append(result_doc)
        
        return results


    # def find_evaluation_texts_full_text(self,query_document,document_type = "text",split_sections = False,**kwargs):
    #     '''
    #     description: for evaluating lucene score threshold

    #     query_document: query document, usually in the form of an entire bill, model legistlation or segment of either
        
    #     document_type: specifies the document type, default: "text" means that know section chunking will be done
    #                     on the query, other options include state bill tuples i.e ("state_bill","al")
    #                     and "model_legislation"

    #     split_sections: specifies whether the query document will be broken into sections to find multiple alignments
    #                     (True) or whether to treat the documents as one and identify a single best alignment (False)
                            
    #     '''

    #     if document_type == "state_bill":
    #         try:
    #             kwargs['state_id']
    #             kwargs['query_document_id'] 
    #         except KeyError:
    #             raise LidException(
    #                     "if document type is state_bill then you musy specify state_id and query_document_id")

    #     elif document_type == "model_legistlation":
    #         try:
    #             kwargs['query_document_id'] 
    #         except KeyError:
    #             raise LidException("if document type is model_legistlation then you musy specify query_document_id")
        
    #     elif document_type == "text":
    #         kwargs['query_document_id'] = None
        
    #     query_document = clean_document(query_document,doc_type = document_type,
    #                 split_to_section = split_sections, **kwargs)
        
    #     elastic_query = u" ".join(query_document)

    #     #query elastic search
    #     result_docs = self.elastic_connection.es_connection.search(q=elastic_query,size = self.results_limit,
    #                                                             index = "evaluation_texts")['hits']['hits']

    #     results = []
    #     for i,result_doc in enumerate(result_docs):

    #         if result_doc['score'] < self.lucene_score_threshold:
    #             break

    #         results.append(result_doc)
        
    #     return results

#Below are functions that use lid objects to identify similar bills/model legislation in the dataset,
#will be moved to another module in the next version

##helper function for process for precompute_bill_similarity
def retrieve_similar_bills(bill_id):
    try:
        ec = database.ElasticConnection()
        bill_doc = ec.get_bill_by_id(bill_id)
        
        bill_text,state = (bill_doc['bill_document_last'],bill_doc['state'])
        
        logging.info("successfully obtained similar docs for {0}".format(bill_id))
        if bill_text is None:
            result_ids = []
        else:
            bill_text = clean_text_for_query(bill_text,state)
            result_docs = ec.similar_doc_query(bill_text,num_results = 10)
            result_ids = [{"id":r['id'],"score":r['score'],"state":r['state']} for r in result_docs]
        
        del ec
        del bill_text
        del state
        return (bill_id,result_ids)
            
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        trace_message = re.sub("\n+", "\t", traceback.format_exc())
        trace_message = re.sub("\s+", " ", trace_message)
        trace_message = "<<{0}>>".format(trace_message)
        m = "Failed to obtain similar docs for {0}: {1}".format(bill_id, trace_message)
        logging.error(m)
        return (bill_id,[])
        
def precompute_bill_similarity(ec):
    """uses elasticsearch queries to find all bill pairs for which there is a potential alignment"""
    
    bill_ids = [x.strip() for x in open("{0}/data/bill_ids.txt".format(os.environ['POLICY_DIFFUSION']))]

    pool = Pool(processes = 7)
    results = pool.map(retrieve_similar_bills,bill_ids)

    return results

def main():
    parser = argparse.ArgumentParser(description='runs scripts for lid system')
    parser.add_argument('command', help='command to run, options are: build_index')
    parser.add_argument('--data_path', dest='data_path', help="file path of data to be indexed ")

    args = parser.parse_args()
    if args.command == "compute_bill_similarity_matrix":
        
        out_file = open("{0}/bill_similarity_matrix.json".format(data_path),'w')
        
        bill_ids = [x.strip() for x in open("{0}/data/bill_ids.txt".format(os.environ['POLICY_DIFFUSION']))]

        pool = Pool(processes = 7)
        results = pool.map(retrieve_similar_bills,bill_ids)
        
        json_obj = {}
        for doc_id,sim_docs in results:
            json_obj[doc_id] = sim_docs
        out_file.write(json.dumps(json_obj))
    

if __name__ == "__main__":
    main()
