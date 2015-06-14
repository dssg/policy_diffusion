#!/usr/bin/env python


from sunlight import openstates
import psycopg2
import csv
import sys
import re



# GRAB DATABASE INFO FROM default_profile
db_info = []
with open('default_profile', 'rb') as db_file:
    reader = csv.reader(db_file, delimiter='=', quotechar='"')
    for row in reader:
        db_info.append(row[1])


# CONNECT TO DATABASE
conn = psycopg2.connect(host = db_info[0], database = db_info[1], user = db_info[2], password = db_info[3]) 
cur = conn.cursor()


# FUNCTION TO PARSE STATE METADATA
def parse_state_metadata(state_metadata):
    name = state_metadata['name']
    abbreviation = state_metadata['abbreviation']
    if 'lower' in state_metadata['chambers']:
        lower_chamber_name = state_metadata['chambers']['lower']['name']
        lower_chamber_title = state_metadata['chambers']['lower']['title']
    else:
        lower_chamber_name = None
        lower_chamber_title = None
    upper_chamber_name = state_metadata['chambers']['upper']['name']
    upper_chamber_title = state_metadata['chambers']['upper']['title']
    feature_flags = ', '.join(state_metadata['feature_flags'])
    return((name, abbreviation, lower_chamber_name, lower_chamber_title,
        upper_chamber_name, upper_chamber_name, feature_flags))


# GRAB THE DATA FROM SUNLIGHT API
state_metadata = openstates.all_metadata()


# PARSE SUNLIGHT DATA AND WRITE TO POSTGRES
temp_state_metadata = []
for state in state_metadata:
    temp_state_metadata.append(parse_state_metadata(state))

args_str = ','.join(cur.mogrify("(%s,%s,%s,%s,%s,%s,%s)", x) for x in temp_state_metadata)
cur.execute("INSERT INTO state_metadata VALUES " + args_str) 
conn.commit()