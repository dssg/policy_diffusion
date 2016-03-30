# This module handles the creation and management of the elasticsearch backend
import os
import argparse
from elasticsearch import Elasticsearch
import json
import time
import base64
from config import DATA_PATH
import utils
from multiprocessing import Pool
import re
import itertools
import numpy as np
import logging
import elasticsearch


#Constants
STATE_BILL_INDEX = "state_bills"
MODEL_LEGISLATION_INDEX = "model_legislation"
EVALUATION_INDEX = "evaluation_texts"
EVALUATION_INDEX_ALL_BILLS = "evaluation_bills_all_bills"


class ElasticConnection():

    def __init__(self,host = "localhost",port = 9200):
        self.es_connection = Elasticsearch([{'host': host, 'port': port}],timeout = 200)

    # creates index for bills and model legislation stored in
    # data_path, overwriting index if it is already created
    def create_state_bill_index(self,data_path):
        if self.es_connection.indices.exists(STATE_BILL_INDEX):
            print("deleting '%s' index..." % (STATE_BILL_INDEX))
            self.es_connection.indices.delete(index=STATE_BILL_INDEX)


        mapping_doc = json.loads(open(os.environ['POLICY_DIFFUSION'] + "/db/state_bill_mapping.json").read())
        settings_doc = json.loads(open(os.environ['POLICY_DIFFUSION'] + "/db/state_bill_index.json").read())

        print("creating '%s' index..." % (STATE_BILL_INDEX))
        res = self.es_connection.indices.create(index=STATE_BILL_INDEX, body=settings_doc)

        print("adding mapping for bill_documents")
        res = self.es_connection.indices.put_mapping(index=STATE_BILL_INDEX, doc_type="bill_document",
                                                body=mapping_doc)

        bulk_data = []
        for i, line in enumerate(open(data_path)):
            json_obj = json.loads(line.strip())
            if json_obj is None:
                continue


            op_dict = {
                "index": {
                    "_index": STATE_BILL_INDEX,
                    "_type": "bill_document",
                    "_id": json_obj["unique_id"]
                }
            }

            bulk_data.append(op_dict)
            bulk_data.append(json_obj)
            if len(bulk_data) == 1000:
                print i
                self.es_connection.bulk(index=STATE_BILL_INDEX, body=bulk_data, timeout=300)

                del bulk_data
                bulk_data = []

    def create_evaluation_index_all_bills(self, data_path1, data_path2):
        '''
        data_path1 corresponds to evaluation data
        data_path2 corresponds to rest of bill data
        '''
        if self.es_connection.indices.exists(EVALUATION_INDEX_ALL_BILLS):
            print("deleting '%s' index..." % (EVALUATION_INDEX_ALL_BILLS))
            self.es_connection.indices.delete(index=EVALUATION_INDEX_ALL_BILLS)

        #use same mapping as in state index 
        mapping_doc = json.loads(open(os.environ['POLICY_DIFFUSION'] + "/db/evaluation_mapping.json").read())
        settings_doc = json.loads(open(os.environ['POLICY_DIFFUSION'] + "/db/state_bill_index.json").read())

        print("creating '%s' index..." % (EVALUATION_INDEX_ALL_BILLS))
        res = self.es_connection.indices.create(index=EVALUATION_INDEX_ALL_BILLS, body=settings_doc,timeout=30)

        print("adding mapping for bill_documents")
        res = self.es_connection.indices.put_mapping(index=EVALUATION_INDEX_ALL_BILLS, doc_type="bill_document",
                                                body=mapping_doc)
    
        #load in evaluation data first
        bulk_data = []
        for i, line in enumerate(open(data_path1)):
            json_obj = json.loads(line.strip())
            if json_obj is None:
                continue

            op_dict = {
                "index": {
                    "_index": EVALUATION_INDEX_ALL_BILLS,
                    "_type": "bill_document",
                    "_id": i
                }
            }

            bulk_data.append(op_dict)
            bulk_data.append(json_obj)
        
        print i
        self.es_connection.bulk(index=EVALUATION_INDEX_ALL_BILLS, body=bulk_data, timeout=300)

        #load in rest of state bill data
        bulk_data = []
        for i, line in enumerate(open(data_path2)):
            json_obj = json.loads(line.strip())
            if json_obj is None:
                continue


            op_dict = {
                "index": {
                    "_index": EVALUATION_INDEX_ALL_BILLS,
                    "_type": "bill_document",
                    "_id": json_obj["unique_id"]
                }
            }

            bulk_data.append(op_dict)
            bulk_data.append(json_obj)
            if len(bulk_data) == 1000:
                print i
                self.es_connection.bulk(index=EVALUATION_INDEX_ALL_BILLS, body=bulk_data, timeout=300)

                del bulk_data
                bulk_data = []

    def create_evaluation_index(self, data_path):
        if self.es_connection.indices.exists(EVALUATION_INDEX):
            print("deleting '%s' index..." % (EVALUATION_INDEX))
            self.es_connection.indices.delete(index=EVALUATION_INDEX)

        #use same mapping as in state index 
        mapping_doc = json.loads(open(os.environ['POLICY_DIFFUSION'] + "/db/evaluation_mapping.json").read())
        settings_doc = json.loads(open(os.environ['POLICY_DIFFUSION'] + "/db/state_bill_index.json").read())

        print("creating '%s' index..." % (EVALUATION_INDEX))
        res = self.es_connection.indices.create(index=EVALUATION_INDEX, body=settings_doc,timeout=30)

        print("adding mapping for bill_documents")
        res = self.es_connection.indices.put_mapping(index=EVALUATION_INDEX, doc_type="bill_document",
                                                body=mapping_doc)

    
        bulk_data = []
        for i, line in enumerate(open(data_path)):
            json_obj = json.loads(line.strip())
            if json_obj is None:
                continue

            op_dict = {
                "index": {
                    "_index": EVALUATION_INDEX,
                    "_type": "bill_document",
                    "_id": i
                }
            }

            bulk_data.append(op_dict)
            bulk_data.append(json_obj)
        
        print i
        self.es_connection.bulk(index=EVALUATION_INDEX, body=bulk_data, timeout=300)

    def get_all_doc_ids(self,index):
        count = self.es_connection.count(index)['count']
        q = {"query":{"match_all" :{} },"fields":[]} 
        results = self.es_connection.search(index = index,body = q,size = count)
        doc_ids = [res['_id'] for res in results['hits']['hits']]
        
        return doc_ids

    
    def similar_doc_query(self,query,state_id = None,num_results = 100,return_fields = ["state"], 
                            index = STATE_BILL_INDEX, fields = "bill_document_last.shingles"):
        json_query = """ 
            {
                "query": {
                    "more_like_this": {
                        "fields": [
                            "%s"
                        ],
                        "like_text": "",
                        "max_query_terms": 25,
                        "min_term_freq": 1,
                        "min_doc_freq": 2,
                        "minimum_should_match": 1
                    }
                }
            }
        """ % (fields)
        json_query = json.loads(json_query)
        json_query['query']['more_like_this']['like_text'] = query


        results = self.es_connection.search(index = index,body = json_query,
                fields = fields,
                size = num_results )
        results = results['hits']['hits']
        result_docs = []
        for res in results:
            doc = {}
            for f in res['fields']:
                doc[f] = res['fields'][f][0]
            #doc['state'] = res['fields']['state'][0]
            doc['score'] = res['_score']
            doc['id'] = res['_id']


            #if applicable, only return docs that are from different states
            if return_fields == ['state']:
                if doc['state'] != state_id:
                    result_docs.append(doc)
            else:
                result_docs.append(doc)
        
        return result_docs

    def similar_doc_query_for_testing_lucene(self,query, match_group, state_id = None,
        num_results = 100,return_fields = ["state"],
        index = STATE_BILL_INDEX):
        '''
        description:
            only for testing lucene scores

        match_group represents the group of bills that an evaluation bill 
        belongs to (e.g., all the stand your ground bills)
        '''

        json_query = """ 
            {
              "query": {
                "filtered": {
                  "query": {
                    "more_like_this": {
                      "fields": [
                        "bill_document_last.shingles"
                      ],
                      "like_text": "",
                      "max_query_terms": 70,
                      "min_term_freq": 1,
                      "min_doc_freq": 2,
                      "minimum_should_match": 1
                    }
                  },
                  "filter": {
                    "bool": {
                      "must_not": {
                        "term": {
                          "bill_document.state": ""
                        }
                      }
                    }
                  }
                }
              }
            }
        """
        json_query = json.loads(json_query)
        json_query['query']['filtered']['query']['more_like_this']['like_text'] = query
        json_query['query']['filtered']['filter']['bool']['must_not']['term']['bill_document.state'] = str(state_id)


        results = self.es_connection.search(index = index,body = json_query,
                fields = return_fields,
                size = num_results )
        results = results['hits']['hits']
        result_docs = []
        for res in results:
            doc = {}
            for f in res['fields']:
                doc[f] = res['fields'][f][0]
            doc['score'] = res['_score']
            doc['id'] = res['_id']
            
            #if applicable, only return docs that are from different states
            if doc['state'] != state_id:
                result_docs.append(doc)
        
        return result_docs


    def get_bill_by_id(self,id, index = 'state_bills'):
        match = self.es_connection.get_source(index = index,id = id)
        return match

        
    def get_all_bills(self, step = 3000):
        es = self.es_connection
        # fix with .format: '{"from" :{0}, "size" :{1}'.format(start,size)
        body_gen = lambda start, size: '{"from" :' + str(start)  + ', "size" : ' + str(size) + ', "query":{"bool":{"must":{"match_all":{}}}}} '
        
        body = body_gen(0,0)
        bills = es.search(index="state_bills", body=body)

        total = bills['hits']['total']

        all_bills = []
        start = 0
        bad_count = 0
        while start <= total:
            print start
            body = body_gen(start,step)
            bills = es.search(index="state_bills", body=body)
            bill_list = bills['hits']['hits']
            all_bills.append(bill_list)
            
            start +=  step

        return all_bills



    def get_bills_by_state(self, state, num_bills = 'all', step = 3000):
        es = self.es_connection

        if num_bills == 'all':
            bills = es.search(index='state_bills', doc_type='bill_document', q= 'state:' + state)
            total = bills['hits']['total']
        else:
            total = num_bills
        
        #fix as above
        body_gen = lambda start, size: '{"from" :' + str(start)  + ', "size" : ' + str(size) +  ',"query":{"term":{"bill_document.state":"' + state+ '"}}}'


        all_bills = []
        start = 0
        bad_count = 0
        if step >= total:
            body = body_gen(start,total)                   
            bills = es.search(index="state_bills", body=body)
            bill_list = bills['hits']['hits']
            all_bills.extend(bill_list)
        else:
            while start <= total:
                body = body_gen(start,step)                   
                bills = es.search(index="state_bills", body=body)
                bill_list = bills['hits']['hits']
                all_bills.extend(bill_list)

                start +=  step

        return all_bills


def parallel_query(query):
    ec = ElasticConnection()
    result = ec.similar_doc_query(query)
    return result


#test to see how query time is affected by speed
def query_time_speed_test():
    from tika import parser as tp
    import re
    import numpy as np

    alec_bills = [json.loads(x) for x in open("{0}/model_legislation/alec_bills.json".format(DATA_PATH))]
    test_queries = [base64.b64decode(s['source']) for s in alec_bills]
    pattern = re.compile("[0-9]\.\s.*") 
    for i,t in enumerate(test_queries):
        test_queries[i] = tp.from_buffer(t)['content']
        test_queries[i] = " ".join(re.findall(pattern,test_queries[i]))
        test_queries[i] = test_queries[i].split()
    
    test_queries = [x for x in test_queries if len(x) >= 1500]
    query_sizes =  np.arange(50,1050,50)
    ec = ElasticConnection()
    avg_times = []
    for query_size in query_sizes:
        temp_times = []
        for query in test_queries:
            query = " ".join(query[0:query_size])
            t1 = time.time()
            ec.similar_doc_query(query,num_results = 1000)
            temp_times.append(time.time()-t1)
        
        avg_times.append(np.mean(temp_times))
        print "query size {0} , avg time (s) {1}".format(query_size,np.mean(temp_times))

    for i in avg_times:
        print i


def parallel_requests_test():
    alec_bills = [json.loads(x) for x in open("{0}/model_legislation/alec_bills.json".format(DATA_PATH))]
    test_queries = [base64.b64decode(s['source']) for s in alec_bills]
    pattern = re.compile("[0-9]\.\s.*") 
    for i,t in enumerate(test_queries):
        test_queries[i] = tp.from_buffer(t)['content']
        test_queries[i] = " ".join(re.findall(pattern,test_queries[i]))
        #test_queries[i] = test_queries[i].split()
        #test_queries[i] = " ".join(test_queries[i][0:200])
    
    test_queries = test_queries[0:100]
    ec = ElasticConnection()
    serial_time = time.time()
    for test_query in test_queries:
        ec.similar_doc_query(test_query)

    print "serial time:  ",time.time()-serial_time
    pool = Pool(processes=7)
    parallel_time = time.time()
    pool.map(parallel_query,test_queries)
    print "parallel time:  ",time.time()-parallel_time
    exit()



## main function that manages unix interface
def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('command', help='command to run, options are: build_index')
    parser.add_argument('--data_path', dest='data_path', help="file path of data to be indexed ")

    args = parser.parse_args()

    if args.command == "build_index":
        ec = ElasticConnection()
        ec.create_state_bill_index(args.data_path)
    elif args.command == "speed_test":
        parallel_requests_test()
        query_time_speed_test()
    
    elif args.command == "test_query":
        ec = ElasticConnection(host = "54.203.12.145",port = "9200")
        ids = ec.get_all_doc_ids("state_bills")
        id_file = open("{0}/data/bill_ids.txt".format(os.environ['POLICY_DIFFUSION']),'w')
        for id in ids:
            id_file.write("{0}\n".format(id))
        
        id_file.close()    

    else:
        print args

def create_evaluation_index():
    state_bill_path = '/mnt/elasticsearch/dssg/extracted_data/extracted_bills.json'
    evaluation_bill_path = '/mnt/data/sunlight/dssg/extracted_data/extracted_evaluation_texts.json'
    es = ElasticConnection()
    es.create_evaluation_index_all_bills(evaluation_bill_path, state_bill_path)

if __name__ == "__main__":
    # main()
    create_evaluation_index()

# b = []
# for key, value in bills.items():
#     b.append({})
#     b[-1]['bill_document_last'] = value['text']
#     b[-1]['match'] = value['match']
#     b[-1]['state'] = value['state']

# outFile = codecs.open("/mnt/data/sunlight/dssg/extracted_data/extracted_evaluation_texts.json", 'w')
# for i, bill in enumerate(b):
#     outFile.write("{0}\n".format(json.dumps(bill)))
# outFile.close()

