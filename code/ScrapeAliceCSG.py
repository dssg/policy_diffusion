'''
webscraper for Alice and CSG websites:

http://www.csg.org/programs/policyprograms/SSL.aspx

https://stateinnovation.org/search/?q=matchall&q.parser=structured&partial=false&return=title%2Csummary%2Cyear%2Csource%2Ctags&size=10&sort=_score%20desc&start=0&fq=(and%20(or%20type_of%3A%20%27Model%20Law%27))&highlight.title=%7B%22pre_tag%22%3A%22%3Cstrong%3E%22%2C%22post_tag%22%3A%22%3C%2Fstrong%3E%22%7D&highlight.summary=%7B%22pre_tag%22%3A%22%3Cstrong%3E%22%2C%22post_tag%22%3A%22%3C%2Fstrong%3E%22%7D
'''

from bs4 import BeautifulSoup
import urllib2
import ujson


#custom packages
import DataExtractor


outDirRootPath = "/mnt/data/sunlight/dssg/scraped_bills/"


#########Alice
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

#store text from links in dictionary to save
aliceText = {}
for link in aliceLinks:
	aliceText[link] = DataExtractor.urlToText(doc)


outFile = open("{0}/{1}".format(outDirRootPath,outFilePath),'w')
outFile.write(ujson.encode(billJson))

# #Get all links from website
# ALEClist = []
# for link in bs.find_all('a'):
#     if link.has_attr('href'):
#        ALEClist.append(link.attrs['href'])
#     