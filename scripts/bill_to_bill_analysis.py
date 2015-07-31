import pandas as pd
import json

with open('/Users/mattburg/Dropbox/bill_similarity_matrix.json') as data_file:
    data = json.load(data_file)

#data = {'ca_1': [{'id': 'ks_2', 'score': 134, 'state': 'ks'}, {'id': 'wy_12', 'score': 80, 'state': 'wy'}],'wa_3': [{'id': 'ca_1', 'score': 20, 'state': 'ca'}, {'id': 'al_5', 'score': 40, 'state': 'al'}]} 


#Need list of dictionary to make it dataframe
df_dict = {}
df_list = []
for item in data:
	for i in range(len(data[item])):
		state_1 = item[0:2]
		state_2 = data[item][i]['state']
		state_1_2 = '-'.join(sorted([state_1, state_2]))
		df_dict={
		'state_1': item[0:2],
		'state_2':data[item][i]['state'],
		'score': data[item][i]['score'],
		'state_1_2': state_1_2}
		df_list.append(df_dict)


df = pd.DataFrame(df_list)

	



