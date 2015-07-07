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

##########################
#Predicate Functions for Cleaning
def isInt(s):
    '''
    Tests whether n can be turned into an integer

    '''
    try:
        int(s)
        return True
    except:
        return False

def isHeaderFooter(s,bill_name):
    '''
    Tests whether s is a header or a footer of the following form:
    bill_name + page_number
    '''
    if re.match(bill_name + ' \d', s) == None:
        return False
    else:
        return True

def isLong(s, n):
    '''
    if a string s contains a word longer than n, it returns True
    '''
    words = s.split()
    for word in words:
        if len(word) >= n:
            return True
    return False

# def isTabs(s):
#     '''
#     checks if line consists only in tabs
#     '''
#     if re.search(r'\t+',s) != None:
#         return True
#     else:
#         return False

def isSpace(s):
    '''
    checks if string consists only of spaces
    '''
    return s.split() == []


def isNewSection(s):
    '''
    Determines whether s is a new section.
    '''
    if re.match(r'^section \d', s) != None:
        return True
    else:
        return False

##########################
#Main clean function

def clean_text(text, bill_name, lower = 1):
    '''
    text:
        text: string corresponding to text of bill
        bill_name: string corresponding to name of the bill

    returns:
        tuple with one field that is string that is cleaned up text 
        and the other field is a dictionary corresponding to sections
    decription:
        clean text and return dictionary with sections
        note that the dictionary of sections only contains material after 
        the first section
    '''
    #parse by line
    text_list = string.split(text,'\n')

    if lower == 1:
        text_list = [text.lower() for text in text_list]

    #replace funky symbols 
    for i in range(len(text_list)):
        text_list[i] = text_list[i].replace(u'\xa0', u' ')

    keep =[]
    prev_newline = 0
    for i in range(len(text_list)):
        line = text_list[i]
        if isLong(line,50) or isInt(line):
            continue
        if prev_newline == 1 and isSpace(line):
            continue
        elif prev_newline == 0 and isSpace(line):
              prev_newline = 1
              keep.append(line)
        else:
            if isHeaderFooter(line, bill_name):
                continue
            else:
                keep.append(line)
                prev_newline = 0


    sections = defaultdict(list)
    seen_section = 0 #have we seen a section yet
    for line in keep:
        if isNewSection(line):
            seen_section = 1
            section = line
            sections[section].append(line)
        elif seen_section == 0:
            continue
        else:
            sections[section].append(line)

    return (string.join(keep, '\n'), sections)



def main():
    es = ElasticConnection(host = "54.212.36.132")

    bills = es.query_state_bills('test')

    for bill in bills:
        bill_name = bill['title']
        (cleaned, sections) = clean_text(bill['doc_text'], bill_name)

        print 'original text: ' + '\n'
        pprint(bill['doc_text'])
        print '\n'
        print 'cleaned text: ' + '\n'
        pprint(cleaned)
        print '\n'
        print 'sections: ' + '\n'
        pprint(sections)
        # print '\n'
        # print 'list: ' + '\n'  
        # pprint(nltk.word_tokenize(cleaned))
        # pprint(cleaned.split())
        raw_input("Press Enter to continue...")


if __name__ == '__main__':
    main()



