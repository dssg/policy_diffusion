import requests
from bs4 import BeautifulSoup
import json
from utils import bill_source_to_json
import urllib2

# Access list of clean urls
with open('/Users/jkatzsamuels/Desktop/dssg/sunlight/policy_diffusion/data/model_legislation_urls/clean_urls.txt',
          'r') as f:
    links = f.read().splitlines()

badCount = 0
goodCount = 0
with open('misc_bills.json', 'w') as jsonfile:
    for link in links:
        try:
            source = urllib2.urlopen(link).read()
            Jsonbill = bill_source_to_json(link, source, None)
            jsonfile.write("{0}\n".format(Jsonbill))
            goodCount += 1
            print goodCount
        except:
            badCount += 1

print str(badCount) + " did not work"
print str(goodCount) + " worked"
