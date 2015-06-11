import requests
from bs4 import BeautifulSoup
import json
from utils import bill_source_to_json

# Access list of clean urls
f = open('/Users/Emily/desktop/sunlight/policy_diffusion/data/model_legislation_urls/clean_urls.txt','r')

# Iterate over list of urls and connect to json file
with open('model-urls.json', 'w') as jsonfile:
    for url in f:
        source = requests.get(url)
        link = url
        # Try to grab the date from each
        date = None
        Jsonbill = bill_source_to_json(link, source, date)
        jsonfile.write("{0}\n".format(Jsonbill))