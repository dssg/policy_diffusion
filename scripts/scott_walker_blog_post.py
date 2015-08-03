import database 



query = open("/Users/mattburg/Dropbox/dssg/policy_diffusion/code/scott_walker_query.txt").read()
ec = database.ElasticConnection(host = "54.203.12.145",port = "9200")

#query elasticsearch
results = ec.similar_doc_query(query,num_results = 100)


out_file = open("/Users/mattburg/Desktop/scott_walker_bills_table.csv",'w')
result_table = []
for res in results:
    id = res['id']
    source_doc = ec.get_bill_by_id(id)
    
    actions = source_doc['action_dates']
    new_actions = []
    for a in actions:
        if actions[a] == None:
            actions[a] = 0
        else:
            actions[a] = 1
    
    
    text = source_doc['bill_document_last']
    if "pain" in text.lower():
        contains_pain = 1
    else:
        contains_pain = 0
        
    result_table.append([id,actions['passed_upper'],actions['passed_lower'],
        actions['signed'],source_doc['date_introduced'],contains_pain])
    
    

out_file.write("id,passed_upper,passed_lower,signed,date_introduced,contains_pain\n")
for r in result_table:
    out_file.write("{0},{1},{2},{3},{4},{5}\n".format(r[0],r[1],r[2],r[3],r[4],r[5]))


