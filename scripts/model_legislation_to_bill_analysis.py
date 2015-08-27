import pandas as pd
import json
from database import *
import numpy as np

#open json file
alec_json = "/Users/eugeniagiraudy/Dropbox/DSSG/policy_diffusion/scripts/model_legislation_alignments.json"

def create_bill_to_bill_matrix(jsonfile):
	'''
	Converts a json file with matching text between model legislation and bills into a 
	dataframe.

	'''
	alignments = [json.loads(x.strip()) for x in open(jsonfile)]
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
	return df


def grab_ids_for_data_frame(df):
	'''
	Grabs bill ids from ElasticSearch and adds it to a dataframe. 
	Outputs csv file with data frame containing model legislation to bills matches and 
	information on date introduced and date signed

	Arguments: 
		dataframe = data frame containing model legislation to bill analysis

	'''
	bill_id_list = df['unique_id']
	bill_id_list = bill_id_list.tolist()

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
	df_dates = pd.DataFrame(bills_introd_signed)
	df_dates.columns = ['unique_id', 'date_introduced', 'date_signed']
	df2 = pd.merge(df, df_dates, on='unique_id')
	#Drop duplicates from the merge
	df3 = df2.drop_duplicates('bill_ml_id')	
	return df3.to_csv('./model_legislation_to_bills_max_score.csv')
	
	


#Analysis of ALEC

df_alec = df3[(df3.interst_group_id =='alec_bills')|(df3.interst_group_id=='alec_old')]
#eliminate cases where two model legislations influence the same bill
df_alec = df_alec.groupby(['unique_id']).max()
date = df_alec['date_introduced']
df_alec['year_introduced']=date.apply(lambda x:x.year)
#eliminate cases wher states may have two identical bills for a given year
df_grouped = df_alec.groupby(['state', 'year_introduced', 'model_legislation_id']).max()
df_grouped.to_csv('./alec_model_legislation_to_bills_max_score_unique.csv')

