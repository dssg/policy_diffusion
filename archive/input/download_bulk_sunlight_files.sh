#!/usr/bin/env bash

### DOWNLOAD BULK DATA ###
eval $(cat /home/jwalsh/policy_diffusion/default_profile | sed 's/^/export /')
state_abbrevs=$(psql -t -c "SELECT abbreviation FROM state_metadata WHERE bills_identified IS NULL AND abbreviation > 'l' ORDER BY abbreviation;")
user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.65 Safari/537.36"
month="07"  # the first day of this month is the last day of records to download
for i in $state_abbrevs; do
	urls="$urls -O http://static.openstates.org/downloads/2016-${month}-01-${i}-json.zip"
done
curl -A '$user_agent' $urls

