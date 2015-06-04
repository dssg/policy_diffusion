"""
Searches google for model legislation

"""

####################
#Compile List of URLs with possible model legislation



from google import search
import pprint
from random import choice

issueList = ["environment", "water", "marijuana", "police brutality"]

urlList = []

for issue in issueList:
	for url in search("model legislation " + issue, stop=20):
		if "alec" not in url: #we don't want model legislation from alec
		    urlList.append(url)


####################
#Find random webpages to train classifier

randomWords= ["hello", "stocks", "politics"] #figure out good way to create list of random words

notBill = []
for i in range(len(randomWords)):
	word1 = choice(randomWords)
	word2 = choice(randomWords)
	for url in search(words1 + " " + word2, stop=10):
		if "alec" not in url: #we don't want model legislation from alec
		    notBill.append(url)

####################
#Grab text from links
