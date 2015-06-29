#Julian Katz-Samuels

'''
run a topic model on bills of each state
'''

from elasticsearch import Elasticsearch
from pprint import pprint
import matplotlib.pyplot as plt


from gensim import corpora, models, similarities
import gensim

import nltk
from nltk.corpus import stopwords

stops = set(stopwords.words('english'))

es = Elasticsearch(hosts = [{"host" : "54.212.36.132", "port" : 9200}], timeout = 300)

body = '{ \
  "size": 0, \
  "aggs": { \
    "states": { \
      "terms": { \
        "field": "state" \
      } \
    } \
  } \
}' 

res = es.search(index="state_bills", body=body)

buckets = res['aggregations']['states']['buckets']
states = []
for i in range(len(buckets)):
	states.append(buckets[i]['key'])

#run topic model on bills of each state
num_topics = 50
models = {} #dictionary of dictionaries containing all relevant information per state
# for state in states:
state = 'ks'

print "working on " + state
models[state] = {}
bills = es.search(index='state_bills', doc_type='bill_document', q= 'state:' + state)
total = bills['hits']['total']
body = '{"size":' + str(total) + ',"query":{"term":{"bill_document.state":"ks"}}}'
bills = es.search(index="state_bills", body=body)
docs = [b['_source']['bill_document_last'] for b in bills['hits']['hits']]


texts = [[word for word in doc.lower().split() if word not in stops] \
          for doc in docs if doc != None]

print "building dictionary and corpus for: " + state
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

print "running lda for: " + state
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=50, update_every=1, chunksize=10000, passes=2)

models[state]['dictionary'] = dictionary
models[state]['corpus'] = corpus
models[state]['model'] = lda


#visualize topic models in various ways
#based on: http://tokestermw.github.io/posts/topicmodel-viz/

#print top ten terms per topic
for i in range(0, num_topics):
    temp = lda.show_topic(i, 10)
    terms = []
    for term in temp:
        terms.append(term)
    print "Top 10 terms for topic #" + str(i) + ": "+ ", ".join([i[1] for i in terms])

#wordcloud
from wordcloud import WordCloud

def terms_to_wordcounts(terms, multiplier=1000):
    return  " ".join([" ".join(int(multiplier*i[0]) * [i[1]]) for i in terms])
 
wordcloud = WordCloud(background_color="black").generate(terms_to_wordcounts(terms))
plt.imshow(wordcloud)


#tranform 
from sklearn.feature_extraction import DictVectorizer
 
def topics_to_vectorspace(n_topics, n_words=100):
    rows = []
    for i in xrange(n_topics):
        temp = lda.show_topic(i, n_words)
        row = dict(((i[1],i[0]) for i in temp))
        rows.append(row)
 
    return rows    
 
vec = DictVectorizer()
 
X = vec.fit_transform(topics_to_vectorspace(num_topics))
X.shape

## PCA
#topics
from sklearn.decomposition import PCA
 
pca = PCA(n_components=2)
 
X_pca = pca.fit(X.toarray()).transform(X.toarray())
 
plt.figure()
for i in xrange(X_pca.shape[0]):
    plt.scatter(X_pca[i, 0], X_pca[i, 1], alpha=.5)
    plt.text(X_pca[i, 0], X_pca[i, 1], s=' ' + str(i))    

plt.title('PCA Topics')
plt.show()

#words
X_pca = pca.fit(X.T.toarray()).transform(X.T.toarray())

plt.figure()
for i, n in enumerate(vec.get_feature_names()):
    plt.scatter(X_pca[i, 0], X_pca[i, 1], alpha=.5)
    plt.text(X_pca[i, 0], X_pca[i, 1], s=' ' + n, fontsize=8)
    
plt.title('PCA Words of Bart Strike Tweets')
plt.show()



