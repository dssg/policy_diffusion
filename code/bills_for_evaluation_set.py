from elasticsearch import Elasticsearch
import re
import csv
import urllib
from urllib import urlopen


es = Elasticsearch(['54.203.12.145:9200', '54.203.12.145:9200'], timeout=300)

def get_bill_by_id(unique_id):
   match = es.search(index="state_bills", body={"query": {"match": {'unique_id': unique_id}}})
   bill_text = match['hits']['hits'][0]['_source']['bill_document_first']
   return bill_text


with open('../data/evaluation_set/bills_for_evaluation_set.csv') as f:
	bills_list = [row for row in csv.reader(f.read().splitlines())]
	
bill_ids_list = []
url_lists = []
topic_list = []
for i in range(len(bills_list)):
	state = bills_list[i][1]
	topic = bills_list[i][0]
	bill_number = bills_list[i][2]
	bill_number = re.sub(' ', '', bill_number)
	year = bills_list[i][3]
	url = bills_list[i][6]
	unique_id = str(state + '_' + year + '_' + bill_number)
	topic_list.append(topic)
	bill_ids_list.append(unique_id)
	url_lists.append(url)

bills_ids = zip(bill_ids_list, url_lists)

bills_text = []
state_list = []
for i in range(len(bills_ids)):
	try:
		bill_text = get_bill_by_id(bills_ids[i][0])
	except IndexError:
		try:
			url = bills_ids[i][1]
			bill_text = urllib.urlopen(url).read()
			print url
		except IOError:
			bill_text = None
	bills_text.append(bill_text)
	state = bills_ids[i][0][0:2]
	state_list.append(state)

bills_state = zip(bills_text, state_list, topic_list)




		


