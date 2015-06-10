#!/usr/bin/env python


import psycopg2
import json
import csv
import os
import re



# GRAB DATABASE INFO FROM default_profile
db_info = []
with open('/home/jwalsh/policy_diffusion/default_profile', 'rb') as db_file:
    reader = csv.reader(db_file, delimiter='=', quotechar='"')
    for row in reader:
        db_info.append(row[1])


# CONNECT TO DATABASE
conn = psycopg2.connect(host = db_info[0], database = db_info[1], user = db_info[2], password = db_info[3]) 
cur = conn.cursor()
        


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



# GRAB BILL METADATA FROM SUNLIGHT AND PUSH TO DATABASE
temp_bill_metadata = []
for path, subdirs, files in os.walk(r'./'):
    for name in files:
        directory_file = os.path.join(path, name)
        if len(temp_bill_metadata) == 10000 or name == files[len(files)-1]:
                args_str = ','.join(cur.mogrify("(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", x) for x in temp_bill_metadata)
                cur.execute("INSERT INTO bill_metadata VALUES " + args_str) 
                conn.commit()
                temp_bill_metadata = []
        with open(directory_file) as json_file:
            bill = json.load(json_file)
            parsed_data = parse_bill_metadata(bill)
            temp_bill_metadata.append(parsed_data)
