#!/usr/env/bin bash

PROFILE:=default_profile
%include $[PROFILE]

#import psycopg2
#import os


state_abbrevs=$(psql -c "SELECT abbreviation FROM input.state_metadata WHERE bills_identified IS NULL ORDER BY abbreviation;" | 
				awk '(NR>2)' |
				head -n -2)

echo $state_abbrevs


# GRAB STATES TRACKED BY SUNLIGHT
#cur.execute("SELECT abbreviation FROM input.state_metadata WHERE bills_identified IS NULL ORDER BY abbreviation;")
#state_abbrev = cur.fetchall()

#print state_abbrev


#/mnt/data/sunlight/openstates_source_files
