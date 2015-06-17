'''
Clean text in ElasticSearch
'''

import elasticsearch
import re
import string

##obtain test data
es = elasticsearch.Elasticsearch(hosts = [{"host" : "54.212.36.132", "port" : 9200}])

q = {"query":{"bool":{"must":[{"query_string":{"default_field":"bill_document.bill_document_last","query":"gun"}}],"must_not":[],"should":[]}},"from":0,"size":10,"sort":[],"facets":{}}

es.search(q)

test = es.get(index = 'state_bills', doc_type = "bill_document", id='ks_2011-2012_HB2011')
test_text = test[u'_source'][u'bill_document_last']

##regex attempt
match = re.findall("^\d+",test_text, re.MULTILINE)

##line-by-line attempt
test_list = string.split(test_text,'\n')

keep =[]
prev_newline = 0
#delete lines with only a number on them and 
#condense multiple empty lines into one empty line
for i in range(len(test_list)):
	line = test_list[i]
	try: 
		int(line)
	except:
		if line != '':
			prev_newline = 0
			keep.append(line)
		elif line == '' and prev_newline == 1:
			continue
		elif line == '' and prev_newline == 0:
			prev_newline = 1
			keep.append(line)

final_text = string.join(keep, '\n')

