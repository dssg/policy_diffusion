"""
Searches google for model legislation

"""

####################
#Compile List of URLs with possible model legislation

from google import search
from bs4 import BeautifulSoup
import urllib2
import pprint
from random import choice

issueList = ["environment", "water", "marijuana", "police brutality"]

urlList = []

for issue in issueList:
	for url in search("model legislation " + issue, stop=3):
		if "alec" not in url: #we don't want model legislation from alec
		    urlList.append(url)


####################
#Find random webpages to train classifier

# issueList= ["hello", "stocks", "politics"] #figure out good way to create list of random words

notBill = []
for i in range(len(issueList)):
	word1 = choice(issueList)
	word2 = choice(issueList)
	for url in search(word1 + " " + word2, stop=3):
		if "alec" not in url: #we don't want model legislation from alec
		    notBill.append(url)


####################
#Grab text from links
urlText = []
for url in urlList:
	try:
		doc = urllib2.urlopen(url).read()
		soup = BeautifulSoup(doc)
		urlText.append(soup.find_all('p'))
	except:
		pass

notBillText = []
for url in notBill:
	try:
		doc = urllib2.urlopen(url).read()
		soup = BeautifulSoup(doc)
		notBillText.append(soup.find_all('p'))
	except:
		pass


##################
#test code




# url = 'http://www.alec.org/model-legislation/72-hour-budget-review-act/'
# html = urllib2.urlopen(url).read()
# soup = BeautifulSoup(html)

# # kill all script and style elements
# for script in soup(["script", "style"]):
#     script.extract()    # rip it out

# # get text
# text = soup.get_text()

# # break into lines and remove leading and trailing space on each
# lines = (line.strip() for line in text.splitlines())
# # break multi-headlines into a line each
# chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
# # drop blank lines
# text = '\n'.join(chunk for chunk in chunks if chunk)

# print(text)



####################
#Train classifier using alec texts and notBills

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


#TODO: read in alec data

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(notBillText)

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#train mode
clf = MultinomialNB().fit(X_train_tfidf, np.zeros((X_train_counts.shape[0],1)))



#test code

# categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
# from sklearn.datasets import fetch_20newsgroups
# twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

# twenty_train.target_names
# len(twenty_train.data)

# from sklearn.feature_extraction.text import CountVectorizer
# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(twenty_train.data)

# #convert to frequencies
# from sklearn.feature_extraction.text import TfidfTransformer
# tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
# X_train_tf = tf_transformer.transform(X_train_counts)

# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts, np.zeros((X_train_counts.shape[0],1))


# from sklearn.naive_bayes import MultinomialNB

# clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)