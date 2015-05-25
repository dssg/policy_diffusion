#!/usr/bin/env bash


state_abbrevs=$(psql -c "\COPY (SELECT abbreviation FROM input.state_metadata WHERE bills_identified IS NULL ORDER BY abbreviation LIMIT 1) TO STDOUT;")

for element in $state_abbrevs; do
	wget -O /mnt/data/sunlight/openstates_zipped_files/$element http://static.openstates.org/downloads/2015-05-01-$element-json.zip
done

#echo $state_abbrevs


# GRAB STATES TRACKED BY SUNLIGHT
#cur.execute("SELECT abbreviation FROM input.state_metadata WHERE bills_identified IS NULL ORDER BY abbreviation;")
#state_abbrev = cur.fetchall()

#print state_abbrev

