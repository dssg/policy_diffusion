'''
Make count of words that appear in final action field.
'''

from elasticsearch import Elasticsearch
from collections import Counter, defaultdict
import numpy as np


es = Elasticsearch(hosts = [{"host" : "54.212.36.132", "port" : 9200}], timeout = 300)


def body_gen(start, size):
	'''
	args:
		from, size ints

	returns:
		string that corresponds to commmand returning all bills starting 
		at number from and a total corresponding to size
	'''
	body = '{"from" :' + str(start)  + ', "size" : ' + str(size) + ', "query":{"bool":{"must":{"match_all":{}}}}} '
	return body

body = body_gen(0,0)
bills = es.search(index="state_bills", body=body)

total = bills['hits']['total']

last_actions = Counter()
b_len = {}
term_counter = Counter()


start = 0
step = 3000
bad_count = 0
while start <= total:
	body = body_gen(start,step)
	bills = es.search(index="state_bills", body=body)
	bill_list = bills['hits']['hits']
	try:
		action_list = [b['_source']['actions'][-1]['action'] for b in bill_list if b['_source']['actions'] != []]
		for action in action_list:
			last_actions[action] += 1

		for bill in bill_list:
			b = bill['_source']
			text = b['bill_document_last']
			b_len[b['sunlight_id']] = {}
			b_len[b['sunlight_id']]['state'] = b['state']
			b_len[b['sunlight_id']]['date'] = b['actions'][0]['date']
			b_len[b['sunlight_id']]['text_length'] = len(b['bill_document_last'].split())
			for word in b['bill_document_last'].split():
				term_counter[word] += 1
	except:
		bad_count += 1
	
	print "counted: " + str(start)
	start +=  step


def isPassed(last_action):
	'''
	Args:
		last_action: string that corresponds to last action of a bill

	Returns:
		True if it satisfies a heuristic for whether it is passed; false otherwise.
	'''

	passed_words = [['approve', 'governor'], ['effective'], ['sign', 'governor'] \
						['enrolled'], ['concur']]

	for words in passed_words:
		passed = True
		for word in words:
			if word not in last_action:
				passed = false
		if passed:
			return True
	return False

status = defaultdict(list)
#Code error here: "list indeces must be integers, not str"
num_passed = Counter()
for key, value in last_actions.iteritems():
	if isPassed(key):
		status['passed'].append(key)
		num_passed['passed'] += value
	else:
		status['failed'].append(key)
		num_passed['failed'] += value	


#plot average document length by month and by state
import matplotlib.pyplot as plt
import pandas as pd
from ggplot import *

df = pd.DataFrame.from_dict(b_len,orient = 'index')
df.date = pd.to_datetime(df.date)

df = df.reset_index()
df = df.set_index('date')
del df['index']
df = df.fillna(value = 0)
df = df.groupby('state').resample('M', how=[np.sum, len])
df = df.fillna(value = 0)

#fix weird indexing issue
df = df.reset_index()
df['sum'] = df['text_length']['sum']
df['num_bills'] = df['text_length']['len']
del df['text_length']
df['avg_bill_length'] = df['sum'] / df['num_bills']
df = df.fillna(value = 0)

#plot average bill length by state and by month
ggplot(aes(x='date', y='avg_bill_length'), df) + \
geom_line(color = 'blue') + facet_wrap('state') +\
ylab('') + \
ggtitle('Average bill length by state') +\
theme(axis_text_x  = element_text(angle = 90, hjust = 1))

#plot number of bills introduced by state and by month

#plot total number of bills introduced by all states vs federal government by month

