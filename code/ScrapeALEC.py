import urllib2
from urllib import urlopen
import bs4
import ujson

url ='http://www.alec.org/model-legislation/'
response = urllib2.urlopen(url).read()
bs = bs4.BeautifulSoup(response, 'html5')

#Get all links from website
ALEClist = []
for link in bs.find_all('a'):
    if link.has_attr('href'):
       ALEClist.append(link.attrs['href'])
    
#Filter list so that we have only the ones with model-legislation
ALEClinks = []
i=0
for i in range(0,len(ALEClist)):
	if ALEClist[i][20:38] == "model-legislation/":
		ALEClinks.append(ALEClist[i])
		i=i+1

#To get only unique links (get rid off duplicates)
ALEClinks = set(ALEClinks)

#Get text and save it to dictionary

ALECdict = {}
i = 0
for line in ALEClinks:
    url_key = {}
    data = urllib2.urlopen(line).read()
    soup = bs4.BeautifulSoup(data)
    ALECtext = soup.findAll("p")
    url_key['url'] = line
    url_key['date'] = 2015
    url_key['entire_html'] = soup
    url_key['model_legislation_text'] = ALECtext
    ALECdict[i] = url_key
    i = i + 1

with open('ALEC_model_legislation.json', 'w') as outfile:
    ujson.dump(ALECdict, outfile)


    




#base64.b64encode(billDocument)



		

       



       
       
      
