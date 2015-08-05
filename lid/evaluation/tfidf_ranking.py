from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
from alignment_evaluation import *

def calc_tfidf(alignments_list):
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
    tfidf = calc_tfidf(alignments_list)

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


if __name__ == '__main__':

    with open('experiment.p', 'rb') as fp:
        e = pickle.load(fp)

    alignments_list = [value['alignments'] for key,value in e.results.iteritems()]

    tfidf = calc_tfidf(alignments_list)

    alignments_tfidf = rank_alignments(alignments_list)
