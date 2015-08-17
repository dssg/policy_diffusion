import lid
import database
from text_alignment import LocalAligner




in_file = "/Users/mattburg/Downloads/LWT_welfare_drug_test.csv"

bill_ids = []
with open(in_file,'rU') as f:
    next(f)
    for line in f:
        if len(line.split(",")[-2]) > 0:
            bill_ids.append(line.split(",")[-2])

    

ec = database.ElasticConnection(host = "54.203.12.145")

for bill_id in bill_ids:
    query = ec.get_bill_by_id(bill_id)['bill_document_last']
    print query
    lidy = lid.LID(query_results_limit=100,lucene_score_threshold = 0.1,
            elastic_host = "54.203.12.145",
            aligner = LocalAligner(match_score = 4, mismatch_score = -1, gap_score = -1))
    result_doc = lidy.find_state_bill_alignments(query,document_type = "state_bill",state_id = bill_id[0:2],
                split_sections = False,query_document_id = bill_id)

    print result_doc.keys()
    print [x['lucene_score'] for x in result_doc['alignment_results']]
    exit()
exit()
lidy = LID(query_results_limit=100,lucene_score_threshold = 0.1,
               )
        
result_doc = lidy.find_state_bill_alignments(model_doc['source'],document_type = "model_legislation",
                split_sections = True,query_document_id = model_doc['id'])


