#!/usr/bin/env bash

source default_profile

rm /mnt/data/sunlight/followthemoney/contributions.csv

for state in AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY
do
	url="http://www.followthemoney.org/aaengine/aafetch.php?s=$state&law-ot=S,H&gro=d-id&APIKey=$FOLLOWTHEMONEYKEY&mode=csv"
	wget -O- --header="Accept: text/html" --user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:21.0) Gecko/20100101 Firefox/21.0" $url >> /mnt/data/sunlight/followthemoney/contributions.csv
done
