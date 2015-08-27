from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
from alignment_evaluation import *
from database import *
import time

def calc_tfidf_alignments(alignments_list):
    '''
	arg:
		list of alignment objects
	returns:
		dictionary with tfi_idf scores
    '''
    corpus = [alignment[1] + alignment[2] \
                for alignments in alignments_list for alignment in alignments ]
    corpus = [' '.join(doc) for doc in corpus]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    idf = vectorizer.idf_
    idf = vectorizer._tfidf.idf_
    return dict(zip(vectorizer.get_feature_names(), idf))


def rank_alignments(alignments_list):
    '''
	arg: 
		list of alignment objects
	returns:
		list of alignment objects sorted by averaged tfi_idf score
    '''
    tfidf = calc_tfidf_alignments(alignments_list)

    not_in_dict = 0
    in_dict = 0

    alignments_tfidf = []
    for alignments in alignments_list:
        tfidf_scores = []
        for alignment in alignments:
            print alignment
            for word in alignment[1]:
                if word in tfidf:
                    tfidf_scores.append(tfidf[word.lower()])
                    in_dict += 1
                if word != '-' and word not in tfidf:
                     not_in_dict += 1
            for word in alignment[2]:
                if word in tfidf:
                    tfidf_scores.append(tfidf[word.lower()])
                    in_dict += 1
                if word != '-' and word not in tfidf:
                     not_in_dict += 1
        if tfidf_scores != []:
            alignments_tfidf.append((alignments, np.sum(tfidf_scores)))
        else:
            alignments_tfidf.append((alignments, 0))

    print "num not in dict: ", not_in_dict
    print "in dict: ", in_dict

    alignments_tfidf.sort(key = lambda x: x[1], reverse=True)

    return alignments_tfidf


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

    def tfidf_score(self, alignment_with_state):
        scores = []
        print 'alignment_with_state: ', alignment_with_state
        raw_input("Press Enter to continue...")
        alignment, left_state, right_state = alignment_with_state
        score, left, right = alignment[0] #TODO: make work for more than one alignment

        for i in range(len(left)):
            scores.append(self.find_tfidf(left[i], left_state)) #need function
            scores.append(self.find_tfidf(right[i], right_state))

        if scores == []:
            return 0
        else:
            return np.mean(scores)


####################################################################
##ranking functions
def rank(alignments_list, functions):
    '''
    depending on the function used, alignments_list may contain states of the alignments of not
    '''
    ranking = []
    for alignments in alignments_list:
        scores = []
        #keep track of for normalization
        max_function_values = np.zeros((4))

        for i in range(len(functions)):
            function = functions[i]
            output = function(alignments)
            scores.append(output)
            ranking.append((alignments, scores))

            if max_function_values[i] < output:
                max_function_values[i] = output

    final_ranking = []
    for alignments, scores in ranking:
        rank_value = []
        scores_max = zip(scores, max_function_values)

        for score, maxim in scores_max:
            rank_value.append(score / float(maxim))

        final_ranking.append((alignments[0][0], np.mean(scores)))

    final_ranking.sort(key = lambda x: x[1], reverse=True)

    return final_ranking


def inspect_ranking(ranking):
    for alignments, tfidf in ranking:
        score, left, right = alignments
        for i in range(len(left)):
            print left[i], right[i]
        print 'alignment score: ', score
        print 'mean tfidf: ', tfidf
        raw_input("Press Enter to continue...")
        print '\n'



if __name__ == '__main__':


    # tfidf = calc_tfidf(alignments_list)

    # alignments_tfidf = rank_alignments(alignments_list)

    # print 'testing speed of calculating tfidf per state'
    
    # t1 = time.time()
    # t=tfidf_state('al')
    # print 'alabama time: {0} seconds'.format(time.time()-t1)

    # t1 = time.time()
    # t=tfidf_state('ny')
    # print 'new york time: {0} seconds'.format(time.time()-t1) 

    # print 'calculate tfidf by state...'

    # tfidf = tfidf_by_all_states()

    # with open('state_tfidfs.p', 'wb') as fp:
    #     pickle.dump(tfidf, fp)

    print 'loading experiment and building alignment list...'
    with open('experiment.p', 'rb') as fp:
        e = pickle.load(fp)

    alignments_list = []
    for key, value in e.results.iteritems():
        i, j = key
        state_i = e.bills[i]['state']
        state_j = e.bills[j]['state']
        alignments_list.append((value['alignments'], state_i, state_j))


    with open('state_tfidfs.p', 'rb') as fp:
        tfidf = pickle.load(fp)
    f = StateTFIDF(tfidf)

    print 'calculating ranking...'
    ranking = rank(alignments_list, [f.tfidf_score])
    inspect_ranking(ranking)

