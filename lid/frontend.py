#!/usr/bin/env python
import os
import sys
import argparse
import datetime as dt
import time
from collections import defaultdict
import cherrypy
from jinja2 import Environment, FileSystemLoader
import random
import string
import json
from elasticsearch import Elasticsearch
from database import ElasticConnection
import re
import nltk
from utils.text_cleaning import clean_document
from lid import LID
from utils.general_utils import alignment_tokenizer
from text_alignment import LocalAligner,AffineLocalAligner



def get_alignment_highlight(text1,text2):
    aligns = align(text1, text2)
    alignment = aligns[0]
    seq1 = nltk.word_tokenize(text1)
    seq2 = nltk.word_tokenize(text2)
    align_clean_1, align_clean_2 = cleanAlignment(alignment)
    [i,j] = contains(align_clean_1, seq1)
    [k,l] = contains(align_clean_2, seq2)
    seq1.insert(i,"<mark>")
    seq1.insert(j,"</mark>")
    seq2.insert(k,"<mark>")
    seq2.insert(l,"</mark>")

    text1  = " ".join(seq1)
    text2 = " ".join(seq2)

    return text1,text2



def markup_alignment_for_display(alignment_dict,left_text,right_text):

    left_text = left_text.split()
    right_text = right_text.split()
    l = alignment_dict['left']
    r = alignment_dict['right']
    left_start = alignment_dict['left_start']
    left_end = alignment_dict['left_end']
    right_start = alignment_dict['right_start']
    right_end = alignment_dict['right_end']



    #mark up l and r alignments with style
    l_styled = []
    r_styled = []
    temp_text = ""
    for i in range(len(l)):
        if l[i] == r[i] and l[i] != "-":
            temp_text+=l[i]
            temp_text+=" "
        if l[i] != r[i]:
            if len(temp_text)>0:
                temp_text = u"<mark>{0}</mark>".format(temp_text) 
                l_styled.append(temp_text)
                r_styled.append(temp_text)
                temp_text = ""
            if l[i] != "-" and r[i] != "-":
                l_styled.append(u"{0}".format(l[i]))
                r_styled.append(u"{0}".format(r[i]))
            else:
                l_styled.append(l[i])
                r_styled.append(r[i])
    
    temp_text = u"<mark>{0}</mark>".format(temp_text) 
    l_styled.append(temp_text)
    r_styled.append(temp_text)

    #l[i] = "<mark>{0}</mark>".format(l[i])
    #r[i] = "<mark>{0}</mark>".format(r[i])

        
    #l.insert(0,"<mark>")
    #l.append("</mark>")
    #r.insert(0,"<mark>")
    #r.append("</mark>")

    


    padding = [u"<br><br>"]

    left_text = left_text[:left_start]+padding+l_styled+\
            padding+left_text[left_end:]

    right_text = right_text[:right_start]+padding+r_styled+padding\
            +right_text[right_end:]
    
    left_text = u" ".join(left_text)
    right_text = u" ".join(right_text)  
    
    return left_text,right_text




def markup_alignment_difference(l,r):
    l_styled = []
    r_styled = []
    temp_text = ""
    for i in range(len(l)):
        if l[i] != r[i]:
            l[i] = u"<mark>{0}</mark>".format(l[i])
            r[i] = u"<mark>{0}</mark>".format(r[i])
     
    return l,r


class DemoWebserver(object):
   

    _cp_config = {
       'tools.staticdir.on' : True,
       'tools.staticdir.dir' : "{0}/html".format(os.environ['POLICY_DIFFUSION']),
       'tools.staticdir.index' : '/templates/searchdemo.html',
       'tools.sessions.on': True,
    }
    
    

    def __init__(self,elastic_connection):
        self.ec = elastic_connection
        self.lidy = LID(elastic_host = "54.203.12.145",query_results_limit=100)
        self.aligner = LocalAligner()
        self.query_bill = "bill"

    @cherrypy.expose
    def searchdemo(self,  query_string = "proof of identity",query_results = []):
        
        query_string =  re.sub('\"',' ',query_string)
        
        query_result = lidy.find_state_bill_alignments(query_string,document_type = "text",
            split_sections = False, query_document_id = "front_end_query" )

        #result_doc_ids = [x['document_id'] for x in query_result['alignment_results']]
        #result_doc_ids = [x.split("_") for x in result_doc_ids]
        #result_doc_ids = [[x[0].upper(),x[1].upper(),x[2]] for x in result_doc_ids]

        results_to_show = []
        for result_doc in query_result['alignment_results']:
            
            meta_data = result_doc['document_id'].split("_")
            meta_data = [meta_data[0].upper(),meta_data[1].upper(),meta_data[2]]
            
            result_text = ec.get_bill_by_id(result_doc['document_id'])['bill_document_last']
            result_text = re.sub('\"',' ',result_text)
            
            alignment = result_doc['alignments'][0]
            score = alignment['score']

            left,right = markup_alignment_for_display(alignment,
                    query_string,result_text)
            left = re.sub('\"',' ',left)
            right = re.sub('\"',' ',right)
            results_to_show.append([score]+meta_data + [left,right])
        
        
        results_to_show.sort(key = lambda x:x[0],reverse = True)
        tmpl = env.get_template('searchdemo.html')
        c = {
                'query_string': query_string,
                'results_to_show': results_to_show,
        }
        return tmpl.render(**c)
    



            
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=29010)
    parser.add_argument('--elasticsearch_connection',default= "localhost:9200")
    args = parser.parse_args()

    env = Environment(loader=FileSystemLoader("{0}/html/templates".format(os.environ['POLICY_DIFFUSION'])))

    query_samples = [x.strip() for x in open("{0}/data/state_bill_samples.txt".format(os.environ['POLICY_DIFFUSION']))]

    aligner = AffineLocalAligner(match_score=4, mismatch_score=-1, gap_start=-3, gap_extend = -1.5)

    ec = ElasticConnection(host = "52.13.123.20")

    lidy = LID(query_results_limit=20,elastic_host = "52.13.123.20",lucene_score_threshold = 0.01,aligner = aligner)
    
    es_host,es_port = args.elasticsearch_connection.split(":") 
    cherrypy.config.update({'server.socket_port': args.port, 'server.socket_host': args.host})
    cherrypy.quickstart(DemoWebserver(ec))
