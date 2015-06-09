'''
webscraper for Alice and CSG websites:

http://www.csg.org/programs/policyprograms/SSL.aspx

https://stateinnovation.org/search/?q=matchall&q.parser=structured&partial=false&return=title%2Csummary%2Cyear%2Csource%2Ctags&size=10&sort=_score%20desc&start=0&fq=(and%20(or%20type_of%3A%20%27Model%20Law%27))&highlight.title=%7B%22pre_tag%22%3A%22%3Cstrong%3E%22%2C%22post_tag%22%3A%22%3C%2Fstrong%3E%22%7D&highlight.summary=%7B%22pre_tag%22%3A%22%3Cstrong%3E%22%2C%22post_tag%22%3A%22%3C%2Fstrong%3E%22%7D
'''

from bs4 import BeautifulSoup
import urllib2
import ujson
import utils


#custom packages
import DataExtractor
from utils import bill_source_to_json

# outDirRootPath = "/mnt/data/sunlight/dssg/scraped_bills/"

#########csg
url ='http://www.csg.org/programs/policyprograms/SSL.aspx'
doc = urllib2.urlopen(url).read()
bs = BeautifulSoup(doc)

aliceLinks = []
for link in bs.find_all('a'):
	if link.has_attr('href'):
		candidate = link.attrs['href']
		if candidate[-4:] == ".pdf": #links with pdf extension tend to be model bills
			aliceLinks.append(candidate) 

#only keeps distinct links
aliceLinks = list(set(aliceLinks))

badCount = 0
goodCount = 0
with open('alice_bills.json', 'w') as f:
	for link in aliceLinks:
	    # url_key = {}
	    # source = urllib2.urlopen(link).read()
	    # Jsonbill = bill_source_to_json(link, source, None)
	    # f.write("{0}\n".format(Jsonbill))		
		try:
		    url_key = {}
		    source = urllib2.urlopen(link).read()
		    Jsonbill = bill_source_to_json(link, source, None)
		    f.write("{0}\n".format(Jsonbill))
		    goodCount += 1
		    print goodCount
		except:
			badCount += 1

print str(badCount) + " did not work"

#store text from links in dictionary to save
# aliceText = {}
# for link in aliceLinks:
# 	aliceText[link] = DataExtractor.urlToText(doc)


# outFile = open("{0}/{1}".format(outDirRootPath,outFilePath),'w')
# outFile.write(ujson.encode(billJson))

#########Alice

#mechanize attempt
import mechanize

br = mechanize.Browser()
# br.set_all_readonly(False)    # allow everything to be written to
# br.set_handle_robots(False)   # ignore robots
# br.set_handle_refresh(False)  # can sometimes hang without this
# # br.addheaders =   	      	# [('User-agent', 'Firefox')]

aliceUrl = 'https://stateinnovation.org/search/?q=matchall&q.parser=structured&partial=false&return=title%2Csummary%2Cyear%2Csource%2Ctags&size=10&sort=_score%20desc&start=0&fq=(and%20(or%20type_of%3A%20%27Model%20Law%27))&highlight.title=%7B%22pre_tag%22%3A%22%3Cstrong%3E%22%2C%22post_tag%22%3A%22%3C%2Fstrong%3E%22%7D&highlight.summary=%7B%22pre_tag%22%3A%22%3Cstrong%3E%22%2C%22post_tag%22%3A%22%3C%2Fstrong%3E%22%7D'
#aliceUrl = 'https://stateinnovation.org/search/'

# response = br.open(aliceUrl)


#url attempt
from bs4 import BeautifulSoup
import requests

doc = requests.get(aliceUrl).text
bs = BeautifulSoup(doc, 'html5')
bs

for link in bs.find_all(''): print link
