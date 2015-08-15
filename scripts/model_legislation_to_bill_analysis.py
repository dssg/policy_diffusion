import pandas as pd
import json
from database import *
import numpy as np

#open json file
alec_json = "/Users/eugeniagiraudy/Dropbox/DSSG/policy_diffusion/scripts/model_legislation_alignments.json"

def create_bill_to_bill_matrix(jsonfile):
	alignments = [json.loads(x.strip()) for x in open(alec_json)]
	df_list = []
	for i in range(len(alignments)):
		left_id = alignments[i]['query_document_id']
		interest_group = left_id.split('_')
		interest_group = "_".join(interest_group[0:2])
		try:
			for result in alignments[i]['alignment_results']:
				right_id = result['document_id']
				score_list = []
				for j in range(len(result['alignments'])):
					score = result['alignments'][j]['score']
					score_list.append(score)
					#Need to decide whehter we want the sum, average, max
					score_max = max(score_list)
				df_list.append([interest_group, left_id,right_id,score_max,right_id[0:2],left_id+"_"+right_id,'undirected'])
		except KeyError:
				print left_id, 'failed'
				continue

	df = pd.DataFrame(df_list)
	df.columns = ['interst_group_id','model_legislation_id', 'unique_id','score_max','state','bill_ml_id','undirected']

#code to create network
	score_mean = df_grouped['score_sum'].tolist()
	index = df_grouped.index
	ids = index.tolist()
	id_list1 = []
	id_list2 = []
	for n in ids:
		id1 = n[:-3]
		id2 = n[-2:]
		id_list1.append(id1)
		id_list2.append(id2)
	final_list = zip(id_list1, id_list2, score_mean)

	
	
	out_file = open("./alec_bill_network.csv",'w')
	for item in final_list:
		out_file.write("{0},{1},{2},{3}\n".format(item[0],item[1],item[2],'undirected'))
	out_file.close()

create_bill_to_bill_matrix(alec_json)

#######

#Get bill_ids to get data on whether the bill passed or not
bill_id_list = df['unique_id']
bill_id_list = bill_id_list.tolist()


#Get info about bills (e.g. dates) using unique_ids and getting information from elasticsearch
ec = ElasticConnection(host = '54.203.12.145', port = 9200)

bill_dates = []
bill_signed = []
for bill in bill_id_list:
    bill_all = ec.get_bill_by_id(bill)
    date_introduced = bill_all['date_introduced']
    date_signed = bill_all['date_signed']
    bill_dates.append(date_introduced)
    bill_signed.append(date_signed)
    print bill
bills_introd_signed = zip(bill_id_list, bill_dates, bill_signed)
out_file = open("./alec_dates.csv", 'w')
for item in bills_introd_signed:
	out_file.write("{0},{1},{2}\n".format(item[0],item[1],item[2]))
out_file.close()


#Make list of dates a df to merge later on
df_dates = pd.DataFrame(bills_introd_signed)
df_dates.columns = ['unique_id', 'date_introduced','date_signed']

#Merge the two data frames and save to csv
df2 = pd.merge(df, df_dates, on='unique_id')
#We had multiple rows for each model_legislation to bill comparison, so I eliminate duplicates using the unique 
#of bill_ml_id
df3 = df2.drop_duplicates('bill_ml_id')


df3_unique = df2.groupby(['unique_id']).max()

df3_unique.to_csv('./alec_model_legislation_to_bills_max_score.csv')





















