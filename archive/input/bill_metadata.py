
import psycopg2
from psycopg2.extras import Json
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
    title = bill_metadata['title']
    if len(bill_metadata['alternate_titles']) > 0:
        alternate_titles = Json(bill_metadata['alternate_titles'][0])
    else:
        alternate_titles = None
    if len(bill_metadata['versions']) > 0:
        versions = Json(bill_metadata['versions'][0])
    else:
        versions = None 
    if 'subjects' in bill_metadata:
        if len(bill_metadata['subjects']) > 0:
            subjects = bill_metadata['subjects'][0]
        else: 
            subjects = None
    else:
        subjects = None
    if 'scraped_subjects' in bill_metadata:
        if len(bill_metadata['scraped_subjects']) > 0:
            scraped_subjects = bill_metadata['scraped_subjects'][0]
        else:
            scraped_subjects = None
    else:
        scraped_subjects = None
    type_ = bill_metadata['type'][0]
    if 'level' in bill_metadata:
        level = bill_metadata['level']
    else:
        level = None
    if len(bill_metadata['sponsors']) > 0:
        sponsors = Json(bill_metadata['sponsors'][0])
    else:
        sponsors = None
    if len(bill_metadata['actions']) > 0:
        actions = Json(bill_metadata['actions'][0])
    else:
        actions = None
    if len(bill_metadata['action_dates']) > 0:
        action_dates = Json(bill_metadata['action_dates'])
    else:
        action_dates = None
    if len(bill_metadata['documents']) > 0:
        documents = Json(bill_metadata['documents'][0])
    else:
        documents = None
    if len(bill_metadata['votes']) > 0:
        votes = Json(bill_metadata['votes'][0])
    else:
        votes = None
    id_ = bill_metadata['id']
    state = bill_metadata['state']
    chamber = bill_metadata['chamber']
    session = bill_metadata['session']
    
    all_ids = bill_metadata['all_ids'][0]
    created_at = bill_metadata['created_at']
    updated_at = bill_metadata['updated_at']

    return((bill_id, title, alternate_titles, versions, subjects, scraped_subjects, 
        type_, level, sponsors, actions, action_dates, documents, votes, id_, state,
        chamber, session, all_ids, created_at, updated_at))



# GRAB BILL METADATA AND PUSH TO DATABASE
temp_bill_metadata = []
for path, subdirs, files in os.walk(r'/mnt/data/sunlight/openstates_unzipped/bills/'):
    for name in files:
        directory_file = os.path.join(path, name)
        with open(directory_file) as json_file:
            bill = json.load(json_file)
            parsed_data = parse_bill_metadata(bill)
            temp_bill_metadata.append(parsed_data)
        if len(temp_bill_metadata) == 10000 or name == files[len(files)-1]:
                args_str = ','.join(cur.mogrify("(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", x) for x in temp_bill_metadata)
                cur.execute("INSERT INTO bill_metadata VALUES " + args_str) 
                conn.commit()
                temp_bill_metadata = []
