#!/usr/bin/env python


from sunlight import openstates
import psycopg2
import re
import csv
import sys



# GRAB DATABASE INFO FROM default_profile
db_info = []
with open('default_profile', 'rb') as db_file:
    reader = csv.reader(db_file, delimiter='=', quotechar='"')
    for row in reader:
        db_info.append(row[1])


# CONNECT TO DATABASE
conn = psycopg2.connect(host = db_info[0], database = db_info[1], user = db_info[2], password = db_info[3]) 
cur = conn.cursor()


# GRAB STATES TRACKED BY SUNLIGHT
cur.execute("SELECT UPPER(abbreviation) FROM input.state_metadata ORDER BY abbreviation;")
state_abbrev = cur.fetchall()


# PARSE BILL METADATA FOR DATABASE INSERTION
def parse_bill_metadata(bill_metadata):
    bill_id = bill_metadata['bill_id']
    chamber = bill_metadata['chamber']
    created_at = bill_metadata['created_at']
    _id = bill_metadata['id']
    session = bill_metadata['session']
    state = bill_metadata['state']
    if 'subjects' in bill_metadata:
        subjects_temp = str(bill_metadata['subjects']).strip('[]')
        subjects = re.sub("u'", "'", subjects_temp)
    else:
        subjects = None
    title = bill_metadata['title']
    _type_temp = str(bill_metadata['type']).strip('[]')
    _type = re.sub("u?'", "", _type_temp)
    updated_at = bill_metadata['updated_at']

    return((bill_id, chamber, created_at, _id, session, state, subjects, title, _type, updated_at))



# GRAB STATE METADATA FROM SUNLIGHT AND PUSH TO DATABASE
for state_meta in state_abbrev: 
    bills = openstates.bills(state=state_meta[0],  window='all')
    temp_bill_metadata = []        
    for bill in bills:
        parsed_data = parse_bill_metadata(bill)
        print parsed_data
        temp_bill_metadata.append(parsed_data)
        if len(temp_bill_metadata) == 1000:
            args_str = ','.join(cur.mogrify("(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", x) for x in temp_bill_metadata)
            cur.execute("INSERT INTO input.bill_metadata VALUES " + args_str) 
            conn.commit()
            temp_bill_metadata = []
