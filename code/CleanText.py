'''
Clean text in ElasticSearch
'''

import elasticsearch
import re
import string
import urllib2
from tika import parser
from collections import defaultdict


#TODO: make the below code into functions that work well with elastic search interface


##obtain test data
# es = elasticsearch.Elasticsearch(hosts = [{"host" : "54.212.36.132", "port" : 9200}])

# q = {"query":{"bool":{"must":[{"query_string":{"default_field":"bill_document.bill_document_last","query":"gun"}}],"must_not":[],"should":[]}},"from":0,"size":10,"sort":[],"facets":{}}

# es.search(q)

# test = es.get(index = 'state_bills', doc_type = "bill_document", id='ks_2011-2012_HB2011')
# test_text = test[u'_source'][u'bill_document_last']

##regex attempt
# match = re.findall("^\d+",test_text, re.MULTILINE)

matches = ['http://www.legislature.mi.gov/(S(ntrjry55mpj5pv55bv1wd155))/documents/2005-2006/billintroduced/House/htm/2005-HIB-5153.htm'
            'http://www.schouse.gov/sess116_2005-2006/bills/4301.htm',
            'http://www.legis.state.wv.us/Bill_Text_HTML/2008_SESSIONS/RS/Bills/hb2564%20intr.html'
            'http://www.lrc.ky.gov/record/06rs/SB38.htm',
            'http://www.okhouse.gov/Legislation/BillFiles/hb2615cs%20db.PDF',
            'https://docs.legis.wisconsin.gov/2011/related/proposals/ab69',
            'http://legisweb.state.wy.us/2008/Enroll/HB0137.pdf',
            'http://www.kansas.gov/government/legislative/bills/2006/366.pdf',
            'http://billstatus.ls.state.ms.us/documents/2006/pdf/SB/2400-2499/SB2426SG.pdf']
nonMatches = ['http://www.alec.org/model-legislation/21st-century-commercial-nexus-act/',
                'http://www.alec.org/model-legislation/72-hour-budget-review-act/',
                'http://www.alec.org/model-legislation/affordable-baccalaureate-degree-act/',
                'http://www.alec.org/model-legislation/agriculture-bio-security-act/',
                'http://www.alec.org/model-legislation/alternative-certification-act/',
                'http://www.alec.org/model-legislation/anti-phishing-act/']

test = nonMatches[2]
doc = urllib2.urlopen(test).read()
test_text = parser.from_buffer(doc)['content']

##line-by-line attempt
test_list = string.split(test_text,'\n')

##########################
#Test Functions
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

def isTabs(s):
    '''
    checks if line consists only in tabs
    '''
    if re.search(r'\t+',s) != None:
        return True
    else:
        return False

def isSpace(s):
    '''
    checks if string consists only of spaces
    '''
    return s.split() == []

##Loop
# keep =[]
# prev_newline = 0
#Cleaning involves:
#1. delete lines with only a number on them and 
#2. condense multiple empty lines into one empty line
#3. remove name
# bill_name = test[u'_source'][u'bill_id']
# for i in range(len(test_list)):
# 	line = test_list[i]
# 	try: 
# 		int(line)
# 	except:
# 		if line != '':
# 			prev_newline = 0
# 			keep.append(line)
# 		elif line == '' and prev_newline == 1:
# 			continue
# 		elif line == '' and prev_newline == 0:
# 			prev_newline = 1
# 			keep.append(line)

# final_text = string.join(keep, '\n')


#replace funky symbols 
for i in range(len(test_list)):
    test_list[i] = test_list[i].replace(u'\xa0', u' ')

# bill_name = test[u'_source'][u'bill_id']
bill_name = 'HB 2011'
keep =[]
prev_newline = 0
for i in range(len(test_list)):
    line = test_list[i]
    if isLong(line,50) or isTabs(line) or isInt(line):
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


final_text = string.join(keep, '\n')


##separate text into sections
#return as dictionary with section as key
#NOTE: that it forgets everything before first section
def isNewSection(s):
    '''
    Determines whether s is a new section.
    '''
    if re.match(r'^Section \d', s) != None:
        return True
    else:
        return False

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



