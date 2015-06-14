
import psycopg2
from psycopg2.extras import Json
import json
import csv
import sys
import re
import os



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
def parse_legislator_metadata(legislator_metadata):
    id_ = legislator_metadata['id']
    if 'votesmart_id' in legislator_metadata:
        votesmart_id = legislator_metadata['votesmart_id']
    else:
        votesmart_id = None
    if 'transparencydata_id' in legislator_metadata:
        transparencydata_id = legislator_metadata['transparencydata_id']
    else:
        transparencydata_id = None
    first_name = legislator_metadata['first_name']
    if len(legislator_metadata['middle_name']) > 0: 
        middle_name = legislator_metadata['middle_name']
    else:
        middle_name = None
    last_name = legislator_metadata['last_name']
    if len(legislator_metadata['suffixes']) > 0:
        suffixes = legislator_metadata['suffixes']
    else:
        suffixes = None
    full_name = legislator_metadata['full_name']
    if 'party' in legislator_metadata:
        party = legislator_metadata['party']
    else:
        party = None
    active = legislator_metadata['active']
    if 'url' in legislator_metadata:
        url = legislator_metadata['url']
    else:
        url = None
    if 'photo_url' in legislator_metadata:
        photo_url = legislator_metadata['photo_url']
    else:
        photo_url = None
    if 'office_address' in legislator_metadata:
        office_address = legislator_metadata['office_address']
    else:
        office_address = None
    if 'office_phone' in legislator_metadata:
        office_phone = legislator_metadata['office_phone']
    else:
        office_phone = None
    leg_id = legislator_metadata['leg_id']
    if 'chamber' in legislator_metadata:
        chamber = legislator_metadata['chamber']
    else:
        chamber = None
    if 'district' in legislator_metadata:
        district = legislator_metadata['district']
    else:
        district = None
    state = legislator_metadata['state']
    if len(legislator_metadata['offices']) > 0:
        offices = Json(legislator_metadata['offices'][0])
    else:
        offices = None
    if 'email' in legislator_metadata:
        email = legislator_metadata['email']
    else:
        email = None
    if len(legislator_metadata['roles']) > 0:
        roles = Json(legislator_metadata['roles'][0])
    else:
        roles = None
    if 'old_roles' in legislator_metadata:
        old_roles = Json(legislator_metadata['old_roles'])
    else:
        old_roles = None
    all_legislative_ids = legislator_metadata['all_ids'][0]
    if 'level' in legislator_metadata:
        level = legislator_metadata['level']
    else:
        level = None
    if len(legislator_metadata['sources']) > 0:
        sources = Json(legislator_metadata['sources'][0])
    else:
        sources = None
    created_at = legislator_metadata['created_at']
    updated_at = legislator_metadata['updated_at']

    return((id_, votesmart_id, transparencydata_id, 
        first_name, middle_name, last_name, suffixes, full_name,
        party, active, url, photo_url, office_address, office_phone,
        leg_id, chamber, district, state, offices, email,
        roles, old_roles, all_legislative_ids, level, sources,
        created_at, updated_at))



# GRAB BILL METADATA FROM SUNLIGHT AND PUSH TO DATABASE
temp_legislator_metadata = []
for path, subdirs, files in os.walk(r'/mnt/data/sunlight/openstates_unzipped/legislators/'):
    for name in files:
        directory_file = os.path.join(path, name)
        with open(directory_file) as json_file:
            legislator = json.load(json_file)
            parsed_data = parse_legislator_metadata(legislator)
            temp_legislator_metadata.append(parsed_data)

args_str = ','.join(cur.mogrify("(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", x) for x in temp_legislator_metadata)
cur.execute("INSERT INTO legislators VALUES " + args_str) 
conn.commit()

