'''
Clean text in ElasticSearch
'''

import elasticsearch
import re
import string
import urllib2
from tika import parser
from collections import defaultdict
from elasticsearch import Elasticsearch
from pprint import pprint
import nltk

#custom modules
from database import ElasticConnection

def clean_text(text, lower = 1):
    '''
    variables:
        text: string corresponding to text of bill
        bill_name: string corresponding to bill_id

    returns:
        string that is cleaned up text 
    decription:
        clean text 
    '''
    #parse by line
    text_list =  text.splitlines()

    #make text lowercase
    if lower == 1:
        text_list = [text.lower() for text in text_list]

    #replace funky symbols and multipe new lines
    ntext_list = []
    for line in text_list:
        line = line.replace(u'\xa0', u' ')
        line = line.replace(u'>>', u' ')
        line = line.replace(u'\xa7', u' ')
        line = line.replace(u'\xe2', u' ')
        line = line.replace(u'\u201c', u' ')
        line = line.replace(u'\u201d', u' ')
        line = line.replace(u'\xbb', u' ')
        line = line.replace(u'\xa9', u' ')
        line = line.replace(u'{ font-family: courier, arial, sans-serif; font-size: 10pt; } table { empty-cells:show; }', u' ')
        line = re.sub( '\s+', ' ', line)
        ntext_list.append(line)
    return (string.join(ntext_list, '\n'))
 
 

#Get data from elasticsearch to test
es = Elasticsearch(['54.203.12.145:9200', '54.203.12.145:9200'], timeout=300)

def test_clean_text(state):
   match = es.search(index="state_bills", body={"query": {"match": {'state': state}}})
   state_text = match['hits']['hits'][3]['_source']['bill_document_first']
   cleantext = clean_text(state_text)
   return cleantext
   

def split_to_sections(cleantext,state):
    '''
    variables:
        cleantext: clean version of text of bill
        state: abbreviation of state ID

    returns:
        list of bill sections
    decription:
        splits bill text into sections
    '''
    if state == 'ak':
        chunked_list = cleantext.split("\n*")
    elif state in ('al','ar','mt','or','ri'):
        chunked_list = cleantext.split('\nsection')
    elif state in ('nm','tx'):
        chunked_list = cleantext.split('\n section')
    elif state in ('az','ia','nv', 'wa'):
        chunked_list = cleantext.split('\nsec.')
    elif state in ('me', 'mi'):
        chunked_list = cleantext.split('\n sec.')
    elif state == 'co':
        chunked_list = re.split('[[0-9][0-9]\.section|[0-9]\.section', cleantext)
    elif state in ('de','fl','tn'):
        chunked_list = re.split('section\s[0-9][0-9]\.|section\s[0-9]\.', cleantext)
    elif state == 'ga':
        cleantext = re.sub('[0-9][0-9]\\n|[0-9]\\n', ' ', cleantext)
        chunked_list = re.split('\\nsection\s[0-9][0-9]|\\nsection\s[0-9]', cleantext)
    elif state in ('hi','sd','in'):
        chunked_list = re.split('\\n\ssection\s[0-9][0-9]\.|\\n\ssection\s[0-9]', cleantext)
    elif state == 'pa':
        chunked_list = re.split('section\s[0-9][0-9]\.|section\s[0-9]\.', cleantext) 
    elif state in ('id', 'la', 'md', 'nd'):
        chunked_list = re.split('\\nsection\s[0-9][0-9]\.|\\nsection\s[0-9]\.', cleantext)
    elif state == 'il':
        cleantext = re.sub('\\n\s[0-9][0-9]|\\n\s[0-9]', ' ', cleantext)
        chunked_list = re.split('\\n\s\ssection\s', cleantext)
    elif state == 'sc':
        chunked_list = cleantext.split('\n \n')
    elif state == 'ks':
        chunked_list = re.split('\\nsection\s|sec\.', cleantext)
    elif state in ('ne', 'mn'):
        chunked_list = re.split('\ssection\s[0-9]\.|\ssec.\s[0-9][0-9]\.|\ssec.\s[0-9]\.', cleantext)
    elif state == 'ky':
        chunked_list = cleantext.split('\n\n\n section .')
    elif state == 'ms':
        chunked_list = cleantext.split('\n\n\n section ')
    elif state in ('ma', 'nc', 'oh','ut'):
        chunked_list = re.split('\ssection\s[0-9][0-9]\.|\ssection\s[0-9]\.', cleantext)
    elif state == 'mo':
        chunked_list = re.split('\\n\s[0-9][0-9]\.\s|\\n\s[0-9]\.\s', cleantext)
    elif state == 'nh':
        chunked_list = re.split('\n\n[0-9][0-9]\s|\n\n[0-9]\s', cleantext)
    elif state == 'nj':
        chunked_list = re.split('\\n\\n\s[0-9][0-9]\.\s|\\n\\n\s[0-9]\.\s', cleantext)
    elif state == 'ny':
        chunked_list = re.split('\ssection\s[0-9]\.|\.\ss\s[0-9]\.', cleantext)
    elif state == 'ok':
        chunked_list = re.split('\nsection\s\.\s', cleantext)
    elif state == 'va':
        chunked_list = re.split('(([A-Z])|[0-9][0-9])\.\s|(([A-Z])|[0-9])\.\s', cleantext)
    elif state == 'wi':
        chunked_list = re.split('\\n[0-9][0-9]section\s\\n|\\n[0-9]section\s\\n', cleantext)
    elif state == 'wv':
        chunked_list = re.split('\n\s\([a-z]\)\s', cleantext)
    elif state == 'wy':
        chunked_list = re.split('\ssection\s[0-9][0-9]\.|\ssection\s[0-9]\.', cleantext)
    elif state == 'ca':
        chunked_list = re.split('section\s[0-9]\.|sec.\s[0-9][0-9]\.|sec.\s[0-9]\.', cleantext)
    else:
        return None

    return chunked_list

#Delete empty sections (run before deleting numbers in lines)
def delete_empty_sections(chunked_list):
    '''
    decription: deletes empty elements in bills
    '''
    return [x for x in chunked_list if x is not None and len(x)>2] 

#Need to delete number lines for: OR, OK, NE, PA (run before deleting lines) 
def delete_numbers_in_lines (chunked_list):
    '''
    decription:
        cleans pdf extractor errors where number of lines were included in text
    '''

    re_string = '\\n\s[0-9][0-9]|\\n[0-9][0-9]|\\n[0-9]|\\n\s[0-9]'
    chunked_list = [re.sub(re_string,'',t) for t in chunked_list]
    return chunked_list


#Delete multiple new lines for each section
def delete_lines (chunked_list):
    '''
    description: deletes multiple lines and spaces for each section
    '''
    chunked_list = [re.sub( '\s+', ' ', x) for x in chunked_list]
    return chunked_list
        

def clean_text_for_query(bill_text,state):
    bill_text = clean_text(bill_text)
    bill_text_sections = split_to_sections(bill_text,state)
    bill_text_sections = delete_empty_sections(bill_text_sections)

    if state in ['or','ok','ne','pa']:
        bill_text_sections = delete_numbers_in_lines(bill_text_sections)
    
    bill_text_sections = delete_lines(bill_text_sections)
    return " ".join(bill_text_sections)


def clean_text_for_alignment(bill_text,state):
    bill_text = clean_text(bill_text)
    bill_text_sections = split_to_sections(bill_text,state)
    bill_text_sections = delete_empty_sections(bill_text_sections)

    if state in ['or','ok','ne','pa']:
        bill_text_sections = delete_numbers_in_lines(bill_text_sections)
    
    bill_text_sections = delete_lines(bill_text_sections)

    return bill_text_sections


def model_clean_text_for_alignment(bill_text):
    bill_text = clean_text(bill_text)

    cleaned_text_list = cleaned_text.split('\n')

    #delete lines with just number
    re_string = '\\n\s[0-9][0-9]|\\n[0-9][0-9]|\\n[0-9]|\\n\s[0-9]'
    cleaned_text_list = [re.sub(re_string,'',t) for t in cleaned_text_list]

    #delete empty lines
    cleaned_text_list = [re.sub( '\s+', ' ', x) for x in cleaned_text_list]
    cleaned_text_list = [x for x in cleaned_text_list if x is not None and len(x)>2]

    cleaned_text = ' '.join(cleaned_text_list)

    return cleaned_text


def test_clean_text_for_alignment(state):
    match = es.search(index="state_bills", body={"query": {"match": {'state': state}}})
    state_text = match['hits']['hits'][3]['_source']['bill_document_first']

    return clean_text_for_alignment(state_text, state)

#good example is test_clean_text_for_alignment('va')
