#!/usr/bin/env bash

eval $(cat ../default_profile | sed 's/^/export /')

state_abbrevs=$(psql -t -c "SELECT abbreviation FROM input.state_metadata WHERE bills_identified IS NULL ORDER BY abbreviation;")

for i in $state_abbrevs; do
	wget -O /mnt/data/sunlight/openstates_zipped_files/${i} http://static.openstates.org/downloads/2015-05-01-${i}-json.zip
done


# GRAB STATES TRACKED BY SUNLIGHT
#cur.execute("SELECT abbreviation FROM input.state_metadata WHERE bills_identified IS NULL ORDER BY abbreviation;")
#state_abbrev = cur.fetchall()

#print state_abbrev

