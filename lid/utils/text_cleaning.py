'''
Clean text in ElasticSearch
'''

import elasticsearch
import re
import string
import urllib2
from elasticsearch import Elasticsearch
from pprint import pprint
import nltk

#custom modules
#from database import ElasticConnection

def clean_text(text, lower = True):
    '''
    variables:
        text: string corresponding to text of bill
        bill_name: string corresponding to bill_id

    returns:
        string that is cleaned up text 
    decription:
        clean text 
    '''
    #make text lowercase
    if lower == True:
        text = text.lower()
    
    #parse by line
    text_list =  text.splitlines()

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
    elif state in ('az','ia','nv', 'wa', 'vt'):
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
    elif state == None:
        chunked_list = cleantext.split("\n")
    else:
        chunked_list = cleantext.split("\n")

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
        


def clean_document(doc_text,doc_type = "text",split_to_section = False,**kwargs):
    """text -- document text
       doc_type --- the type of the document ( "state_bill", "model_legislation", "None")    """
    
    if doc_type == "state_bill":
        doc_text = clean_text(doc_text)
        doc_text_sections = split_to_sections(doc_text,kwargs['state_id'])
        doc_text_sections = delete_empty_sections(doc_text_sections)
        if kwargs['state_id'] in ['or','ok','ne','pa']:
            doc_text_sections = delete_numbers_in_lines(doc_text_sections)
        doc_text_sections = delete_lines(doc_text_sections)
    
    elif doc_type == "model_legislation":
        doc_text = clean_text(doc_text)
        doc_text_sections = doc_text.split('\nsection')
        doc_text_sections = delete_empty_sections(doc_text_sections)
        doc_text_sections = delete_lines(doc_text_sections)
        
    elif doc_type == "text":
        doc_text = clean_text(doc_text)
        doc_text_sections = doc_text.split('\n')
        doc_text_sections = delete_empty_sections(doc_text_sections)
        doc_text_sections = delete_lines(doc_text_sections)
    
    if split_to_section == True:
        return doc_text_sections
    elif split_to_section == False:
        return [" ".join(doc_text_sections)]

#delete boiler plate present in all alec exposed bills after "effective date"
def delete_boiler_plate_alec_exposed (chunked_list):
    chunked_list = [re.sub('({effective date).*$', ' ', x) for x in chunked_list]
    chunked_list = chunked_list[1:]
    return chunked_list

#good example is test_clean_text_for_alignment('va')

def test_clean_text(state):
   es = Elasticsearch(['54.203.12.145:9200', '54.203.12.145:9200'], timeout=300)
   match = es.search(index="state_bills", body={"query": {"match": {'state': state}}})
   state_text = match['hits']['hits'][3]['_source']['bill_document_first']
   cleaned_doc = clean_document(state_text,doc_type = "state_bill",state_id = "mi",split_to_section = False)
   return cleaned_doc

def main():
    #Get data from elasticsearch to test
    
    print test_clean_text("mi")

if __name__ == "__main__":
    main()




