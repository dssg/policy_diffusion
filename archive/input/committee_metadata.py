
from sunlight import openstates
import psycopg2
from psycopg2.extras import Json
import json
import csv
import sys
import re
import os


# GRAB DATABASE INFO FROM default_profile
db_info = []
with open('default_profile', 'rb') as db_file:
    reader = csv.reader(db_file, delimiter='=', quotechar='"')
    for row in reader:
        db_info.append(row[1])


# CONNECT TO DATABASE
conn = psycopg2.connect(host = db_info[0], database = db_info[1], user = db_info[2], password = db_info[3]) 
cur = conn.cursor()


# PARSE COMMITTEE METADATA
def parse_committee_metadata(committee_metadata):
    id_ = committee_metadata['id']
    state = committee_metadata['state']
    chamber = committee_metadata['chamber']
    committee = committee_metadata['committee']
    subcommittee = committee_metadata['subcommittee']
    if len(committee_metadata['members']) > 0:
        members = Json(committee_metadata['members'][0])
    else: 
        members = None
    sources = committee_metadata['sources'][0]['url']
    parent_id = committee_metadata['parent_id']
    created_at = committee_metadata['created_at']
    updated_at = committee_metadata['updated_at']
    if len(committee_metadata['all_ids']) > 0:
        all_ids = committee_metadata['all_ids'][0]
    else:
        all_ids = None
    if 'level' in committee_metadata:
        level = committee_metadata['level']
    else:
        level = None

    return((id_, state, chamber, committee, subcommittee, members,
        sources, parent_id, created_at, updated_at, all_ids, level))



# GRAB COMMITTEE METADATA FROM FILES AND PUSH TO DATABASE
temp_committee_metadata = []
for path, subdirs, files in os.walk(r'/mnt/data/sunlight/openstates_unzipped/committees/'):
    for name in files:
        directory_file = os.path.join(path, name)
        with open(directory_file) as json_file:
            committee = json.load(json_file)
            parsed_data = parse_committee_metadata(committee)
            temp_committee_metadata.append(parsed_data)

args_str = ','.join(cur.mogrify("(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", x) for x in temp_committee_metadata)
cur.execute("INSERT INTO committees VALUES " + args_str) 
conn.commit()

