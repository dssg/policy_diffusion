#!/usr/bin/env python
import os
import sys
import argparse
import datetime as dt
import time
from collections import defaultdict
from fast_alignment import LocalAligner
import cherrypy
from jinja2 import Environment, FileSystemLoader
import random
import string
import json
from elasticsearch import Elasticsearch
from database import ElasticConnection
import re
import nltk
from CleanText import clean_text_for_query
from LID import LID


env = Environment(loader=FileSystemLoader('/Users/mattburg/Dropbox/dssg/policy_diffusion/html/templates'))

#EVALUATION_DATA = json.load(open("../data/eval.json")).values()
#EVALUATION_DATA = [[d['url'],d['text'],d['match'],i] for i,d in enumerate(EVALUATION_DATA) if len(d.values()) > 0]
#EVALUATION_DATA.sort(key = lambda x:x[2])

#for d in EVALUATION_DATA:
#    d[0] = "/".join(d[0].split("/")[0:4])
#    text = d[1]
#    text = re.sub("\n+","\n",text)
#    text = re.sub('\"',' ',text)
#    d[1] = text



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
       'tools.staticdir.dir' : '/Users/mattburg/Dropbox/dssg/policy_diffusion/html',
       'tools.staticdir.index' : 'index.html',
       'tools.sessions.on': True,
    }
    
    

    def __init__(self,elastic_connection):
        self.ec = elastic_connection


    @cherrypy.expose
    def searchdemo(self, query_string="this is a test"):

        query_string =  re.sub('\"',' ',query_string)

                
        query_string = clean_text_for_query(query_string,state = None)
        query_results = self.ec.query_state_bills_for_frontend(query_string)
        query_results = [[r['text'],r['score'],r['id'],r['state']] for r in query_results]
        
        
        alignment_results = []
        for q in query_results:
            text = q[0]
            text = re.sub('\"',' ',text)
            text = clean_text_for_query(text,state = None)
            query_list = query_string.split()
            text = text.split()
            f = LocalAligner()
            alignments=f.align(query_list,text) #default score is 3,-1,-2
            score, l, r  = alignments.alignments[0]
            
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
                        temp_text = "<mark>{0}</mark>".format(temp_text) 
                        l_styled.append(temp_text)
                        r_styled.append(temp_text)
                        temp_text = ""
                    if l[i] != "-" and r[i] != "-":
                        l_styled.append(u"<b>{0}</b>".format(l[i]))
                        r_styled.append(u"<b>{0}</b>".format(r[i]))
                    else:
                        l_styled.append(l[i])
                        r_styled.append(r[i])
            
            temp_text = "<mark>{0}</mark>".format(temp_text) 
            l_styled.append(temp_text)
            r_styled.append(temp_text)

                    #l[i] = "<mark>{0}</mark>".format(l[i])
                    #r[i] = "<mark>{0}</mark>".format(r[i])

                
            #l.insert(0,"<mark>")
            #l.append("</mark>")
            #r.insert(0,"<mark>")
            #r.append("</mark>")

            


            padding = ["<br><br>------------------------------MATCH-------------------------------<br><br>"]

            left_text = query_list[:f.alignment_indices[0]['left_start']]+padding+l_styled+\
                    padding+query_list[f.alignment_indices[0]['left_end']:]

            right_text = text[:f.alignment_indices[0]['right_start']]+padding+r_styled+padding\
                    +text[f.alignment_indices[0]['right_end']:]
            
            left_text = " ".join(left_text)
            right_text = " ".join(right_text)    
            
            alignment_results.append( [ left_text,right_text,"{0:.2f}".format(q[1]),q[2]])
            
            
        tmpl = env.get_template('searchdemo.html')
        c = {
                'query_string': query_string,
                'query_results': alignment_results,
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
    ec = ElasticConnection(es_host,es_port)
    cherrypy.config.update({'server.socket_port': args.port, 'server.socket_host': args.host})
    cherrypy.quickstart(DemoWebserver(ec))
