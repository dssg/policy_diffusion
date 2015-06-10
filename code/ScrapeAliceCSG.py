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



#########Alice

#Already made files with links in them; now need to extract them
path = "/Users/jkatzsamuels/Desktop/dssg/sunlight/test_code/csg_files/links_"

lines = []
for i in [1,2,3]:
	filePath = path + str(i) + ".txt"
	with open(filePath) as f:
		lines.extend(f.read().splitlines())

text = ''.join(lines)
bs = BeautifulSoup(text)

links = []
for link in bs.find_all('a'):
	if link.has_attr('href'):
		links.append(link.attrs['href'])


#grab pdfs from links
billList = []
for url in links:
	doc = urllib2.urlopen(url).read()
	bs = BeautifulSoup(doc)

	for link in bs.find_all('a'):
		if link.has_attr('href'):
			candidate = link.attrs['href']
			if candidate[-4:] == ".pdf": #links with pdf extension tend to be model bills
				billList.append("https://stateinnovation.org" + candidate) 


badCount = 0
goodCount = 0
with open('csg_bills.json', 'w') as f:
	for link in billList:
	    # url_key = {}
	    # source = urllib2.urlopen(link).read()
	    # Jsonbill = bill_source_to_json(link, source, None)
	    # f.write("{0}\n".format(Jsonbill))		
		try:
		    source = urllib2.urlopen(link).read()
		    Jsonbill = bill_source_to_json(link, source, None)
		    f.write("{0}\n".format(Jsonbill))
		    goodCount += 1
		    print goodCount
		except:
			badCount += 1

print str(badCount) + " did not work"



