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
from alignmentFunctions import align,cleanAlignment,contains
import re
import nltk


env = Environment(loader=FileSystemLoader('/Users/eugeniagiraudy/Dropbox/DSSG/policy_diffusion/html/templates'))

EVALUATION_DATA = json.load(open("../data/eval.json")).values()
EVALUATION_DATA = [[d['url'],d['text'],d['match'],i] for i,d in enumerate(EVALUATION_DATA) if len(d.values()) > 0]
EVALUATION_DATA.sort(key = lambda x:x[2])

for d in EVALUATION_DATA:
    d[0] = "/".join(d[0].split("/")[0:4])
    text = d[1]
    text = re.sub("\n+","\n",text)
    text = re.sub('\"',' ',text)
    d[1] = text



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

    




class DemoWebserver(object):
   

    _cp_config = {
       'tools.staticdir.on' : True,
       'tools.staticdir.dir' : '/Users/eugeniagiraudy/Dropbox/DSSG/policy_diffusion/html',
       'tools.staticdir.index' : 'index.html',
       'tools.sessions.on': True,
    }
    
    

    def __init__(self,elastic_connection):
        self.ec = elastic_connection


    @cherrypy.expose
    def searchdemo(self, query_string="this is a test",result_type = "raw_text"):
                
        
        query_results = self.ec.query_state_bills(query_string)
        
        
        if result_type == "raw_text":
            query_results = [[r['doc_text_with_highlights'],r['score'],r['title']] for r in query_results]
            #format text 
            for q in query_results:
                text = q[0][0]
                text = re.sub("\n+","\n",text)
                text = re.sub('\"',' ',text)
                q[0] = text
                q[1] = "{0:.2f}".format(q[1])
                q[2] = re.sub("\n"," ",q[2])
                q[2] = q[2][0:80].lower()
        
        elif result_type == "alignment":
            query_results = [[r['doc_text'],r['score'],r['title']] for r in query_results]
            query_results = query_results[0:1]
            for q in query_results:
                text = q[0]
                text = re.sub("\n+","\n",text)
                text = re.sub('\"',' ',text)
                alignment = align(query_string,text)
                print "type", " ".join(alignment[0][1])
                print "type 2"," ".join(alignment[0][2])
                q[0] = u"alignment:  \n{0}\n\n{1}\n\n{2}".format(" ".join(alignment[0][1])," ".join(alignment[0][2]),text)
                print q[0]
                q[1] = "{0:.2f}".format(q[1])
                q[2] = re.sub("\n"," ",q[2])
                q[2] = q[2][0:80].lower()




        tmpl = env.get_template('searchdemo.html')
        c = {
                'query_string': query_string,
                'query_results': query_results,
                'result_type': result_type
        }
        return tmpl.render(**c)


    @cherrypy.expose
    def alignmentdemo(self, evaluation_data = None,left_doc_id = None,right_doc_id = None ):
                
        if left_doc_id != None and right_doc_id != None:   
            doc_left = EVALUATION_DATA[int(left_doc_id)][1]
            doc_right = EVALUATION_DATA[int(right_doc_id)][1]
            doc_left,doc_right = get_alignment_highlight(doc_left,doc_right)
            
            
            tmpl = env.get_template('alignmentdemo.html')
            c = {
                    
                    'evaluation_data': EVALUATION_DATA,
                    'doc_left': doc_left,
                    'doc_right': doc_right,

            }
            return tmpl.render(**c)
        else:
            tmpl = env.get_template('alignmentdemo.html')
            c = {
                    
                    'evaluation_data': EVALUATION_DATA,

            }
            return tmpl.render(**c)

            
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=29010)
    parser.add_argument('--elasticsearch_connection',default= "localhost:9200")
    args = parser.parse_args()
    
    es_host,es_port = args.elasticsearch_connection.split(":") 
    ec = ElasticConnection(es_host,es_port )
    cherrypy.config.update({'server.socket_port': args.port, 'server.socket_host': args.host})
    cherrypy.quickstart(DemoWebserver(ec))
