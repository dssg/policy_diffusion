'''
Functions for scoring alignments
'''

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from database import *

def weight_length(alignment, left_length, right_length):
	print alignment
	return np.sum([a[0]*(len(a[1])/float(left_length))*(len(a[2])/float(right_length)) for a in alignment.alignments])

def weight_tfidf(alignment, state_tfidf, left_state, right_state):
	'''
	state_tfidf: dictionary with tfidf scores by state
	'''
	f = StateTFIDF(state_tfidf)
	return np.sum([f.tfidf_score(a, left_state, right_state)*a[0] for a in alignment.alignments])

####################################################################
##tfidf functions

def tfidf_by_state(state, num_bills = 'all'):
    '''
    description:
        create dictionary of tfidf scores for a particular
    args:
        state
        num_bills: number of bills to run the algorithm open
    returns:
        dictionary of tfidf scores with words as keys
    '''
    es = ElasticConnection()
    state_bills = es.get_bills_by_state(state, num_bills)
    corpus = [bill['_source']['bill_document_last'] for bill in state_bills \
            if bill['_source']['bill_document_last'] != None]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    idf = vectorizer.idf_
    idf = vectorizer._tfidf.idf_

    return dict(zip(vectorizer.get_feature_names(), idf))


def tfidf_by_all_states():
    states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 
            'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO','MT', 'NE', 
            'NV', 'NH','NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 
            'TN', 'TX', 'UT', 'VT',  'VA', 'WA', 'WV', 'WI', 'WY']
    states = map(lambda x : x.lower(), states)

    tfidf = {}
    for state in states:
        print 'working on ', state
        tfidf[state] = tfidf_by_state(state)

    return tfidf


####################################################################
##state tfidf object
class StateTFIDF():

    def __init__(self, state_tfidf):
        self.state_tfidf = state_tfidf

    def find_tfidf(self, word, state):
        if state == 'model_legislation':
            return 0
        elif word == '-' or word not in self.state_tfidf[state]:
            return 0
        else:
            return self.state_tfidf[state][word]

    def tfidf_score(self, alignment, left_state, right_state):
    	'''
    	gives average tfidf for a particular alignment
    	'''
        scores = []
        score, left, right = alignment 

        for i in range(len(left)):
            scores.append(self.find_tfidf(left[i], left_state)) #need function
            scores.append(self.find_tfidf(right[i], right_state))

        if scores == []:
            return 0
        else:
            return np.mean(scores)
