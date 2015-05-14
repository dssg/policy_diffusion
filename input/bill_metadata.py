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
cur.execute


# GRAB STATE METADATA FROM SUNLIGHT
bill_metadata = openstates.bills()


# PARSE SUNLIGHT DATA AND WRITE TO STDOUT 
a = csv.writer(sys.stdout)
for state in state_metadata:
    abbreviation = state['abbreviation']
    if 'lower' in state['chambers']:
        lower_chamber_name = state['chambers']['lower']['name']
        lower_chamber_title = state['chambers']['lower']['title']
    else:
        lower_chamber_name = None
        lower_chamber_title = None
    upper_chamber_name = state['chambers']['upper']['name']
    upper_chamber_title = state['chambers']['upper']['title']
    feature_flags = str(state['feature_flags']).strip('[]')
    a.writerow((abbreviation, lower_chamber_name, lower_chamber_title, upper_chamber_name, upper_chamber_title, feature_flags))
