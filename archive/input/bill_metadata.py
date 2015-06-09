#!/usr/bin/env python


import psycopg2
import os


host = os.environ.get('PGHOST')
db = os.environ.get('PGDATABASE=')
user = os.environ.get('PGUSER')
passwd = os.environ.get('PGPASSWORD')

conn = psycopg2.connect(host = host, database = db, user = user, password = passwd) 
cur = conn.cursor()


# GRAB STATES TRACKED BY SUNLIGHT
cur.execute("SELECT abbreviation FROM input.state_metadata WHERE bills_identified IS NULL ORDER BY abbreviation;")
state_abbrev = cur.fetchall()

print state_abbrev


# PARSE BILL METADATA FOR DATABASE INSERTION
def parse_bill_metadata(bill_metadata):
    bill_id = bill_metadata['bill_id']
    chamber = bill_metadata['chamber']
    created_at = bill_metadata['created_at']
    id_ = bill_metadata['id']
    session = bill_metadata['session']
    state = bill_metadata['state']
    if 'subjects' in bill_metadata:
        if bill_metadata['subjects']:
            subjects_temp = str(bill_metadata['subjects']).strip('[]')
            subjects = re.sub("u'", "'", subjects_temp)
        else:
            subjects = None
    else:
        subjects = None
    title = bill_metadata['title']
    type_temp = str(bill_metadata['type']).strip('[]')
    type_ = re.sub("u?'", "", type_temp)
    updated_at = bill_metadata['updated_at']

    return((bill_id, chamber, created_at, id_, session, state, subjects, title, type_, updated_at))



# GRAB STATE METADATA FROM SUNLIGHT AND PUSH TO DATABASE
for state_meta in state_abbrev[0]: 
    print state_meta
    
    bills = openstates.bills(state=state_meta,  window='term')
    temp_bill_metadata = []        
    for bill in bills:
        parsed_data = parse_bill_metadata(bill)
        temp_bill_metadata.append(parsed_data)
        if len(temp_bill_metadata) == 1000 or parsed_data[3] == bills[len(bills)-1]['id']:
            args_str = ','.join(cur.mogrify("(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", x) for x in temp_bill_metadata)
            cur.execute("INSERT INTO input.bill_metadata VALUES " + args_str) 
            conn.commit()
            temp_bill_metadata = []
    
    update_statement = "UPDATE input.state_metadata SET bills_identified = TRUE WHERE abbreviation = '%s';" % (state_meta)
    cur.execute(update_statement)
    conn.commit()

