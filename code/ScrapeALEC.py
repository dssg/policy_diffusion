import urllib2
from urllib import urlopen
import bs4
import ujson
from utils import bill_source_to_json
from os import listdir
from DataExtractor import pdfToText

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

#Save to json file
with open('alec_bills.json', 'w') as f:
    for line in ALEClinks:
        source = urllib2.urlopen(line).read()
        url = line
        date = 2015
        Jsonbill = bill_source_to_json(url, source, date)
        f.write("{0}\n".format(Jsonbill))


##Old ALEC urls
path = '../../PolicyDiffusion/ALEC_exposed'
pdf_source = pdfToText('../../PolicyDiffusion/ALEC_exposed/Paycheck Protection Act Exposed.pdf')
names = ('../../PolicyDiffusion/ALEC_exposed/7J4-Concealed_Carry_True_Reciprocity_Act Exposed.pdf')
for name in listdir(path):
    pdf_source = pdfToText(name)
    print(pdf_source)



