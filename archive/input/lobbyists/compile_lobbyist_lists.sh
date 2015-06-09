#!/usr/bin/env bash


## ILLINOIS ##
#url=


## MICHIGAN ##
# http://miboecfr.nictusa.com/cgi-bin/cfr/lobby_srch_res.cgi

url=http://miboecfr.nictusa.com/cfr/dumpdata/aaarZaGrk/mi_lobby.sh
wget -O michigan_lobbyists.txt --user-agent="jtwalsh@uchicago.edu" $url 

#sed -E 's/\t/,/g' michigan_lobbyists.csv | sed 's/#/ Number/g' | sed -E 's/\(MaxLen=(.){1,3}\)//g'

http://miboecfr.nictusa.com/cfr/dumpdata/aaa3AaiZp/mi_lobby.sh

# second line of the file has metadata
# the bottom of the file has garbage too

