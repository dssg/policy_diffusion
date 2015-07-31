from __future__ import division
import numpy as np
from numba import jit
import itertools
import time
import matplotlib.pyplot as plt
import sys
import abc
import json
import pickle
from tika import parser
import urllib2
import urllib
from urllib import urlopen
import re
import pandas as pd
from sklearn.decomposition import PCA
from fast_alignment import *    
from CleanText import clean_text_for_alignment, clean_text_for_query
from compiler.ast import flatten
from elasticsearch import Elasticsearch
import re
import csv
from sklearn.metrics import roc_curve, auc


#TODO: incorporate plotting into grid code; not sure how to do this in way that works for all algorithms

class Experiment():

    def __init__(self, bills, algorithm, match_score = 3, mismatch_score = -1, 
                gap_score = -2, gap_start = -5, gap_extend = -0.5):
    	self.bills = bills
    	self.algorithm = algorithm
        if bills == {}:
            self.scores = None
        else:
    	   self.scores = np.zeros((max(self.bills.keys())+1, max(self.bills.keys())+1))
    	self.results = {}
        self.match_score = match_score
        self.mismatch_score = mismatch_score
        self.gap_score = gap_score
        self.gap_start = gap_start
        self.gap_extend = gap_extend


    def evaluate(self):
        self.evaluate_algorithm(self.bills)

        self.plot_scores(self.scores, self.bills) 

        return self.scores, self.results


    def evaluate_algorithm(self):
        '''
        args:
            matches: dictionary with field corresponding to text and match groups

        returns:
            matrix with scores between all pairs and a dictionary with information
        '''
        for i in self.bills.keys():
            for j in self.bills.keys():  
                if i != j:              

                    if self.bills[i] == {} or self.bills[j] == {}:
                        continue

                    if bills[i]['text'] == '' or bills[j]['text'] == '':
                        continue

                    text1 = self._prepare_text_left(bills[i]['text'], bills[i]['state'])
                    text2 = self._prepare_text_right(bills[j]['text'], bills[j]['state'])

                    #instantiate aligner with appropriate parameters
                    if self.algorithm == LocalAligner:
                        f = self.algorithm(self.match_score, self.mismatch_score, self.gap_score)
                    elif self.algorithm == AffineLocalAligner:
                        f = self.algorithm(self.match_score, self.mismatch_score, self.gap_start, self.gap_extend)
                    alignment = f.align(text1, text2)

                    self.scores[i,j] = self._get_score(alignment)

                    self.results[(i,j)] ={}
                    self.results[(i,j)]['alignments'] = alignment.alignments
                    self.results[(i,j)]['score'] = self._get_score(alignment)
                    self.results[(i,j)]['match'] = (bills[i]['match'] == bills[j]['match'])
                    self.results[(i,j)]['diff'] = [self._diff(a) for a in alignment.alignments]
                    self.results[(i,j)]['features'] = [self._alignment_features(a[1],a[2]) for a in alignment.alignments]

                    print 'i: ' + str(i) + ', j: ' + str(j) + ' score: ' + str(alignment.alignments[0][0])

        return self.scores, self.results

    def plot_roc():
        truth = [value['match'] for key, value in self.results.items()]
        score = [value['score'] for key, value in self.results.items()]

        fpr, tpr = roc_curve(truth, score)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()


    def save(self):
        if self.algorithm == LocalAligner:
            with open('../data/experiment_results/experiment_{0}_m_{1}_mm_{2}_g_{3}.p'.format(self.algorithm._algorithm_name, 
                                self.match_score, self.mismatch_score, self.gap_score), 'wb') as fp:
                
        pickle.dump(self, fp)

    @abc.abstractmethod
    def _prepare_text_left(self):
    	pass


    @abc.abstractmethod
    def _prepare_text_right(self):
        pass


    @abc.abstractmethod
    def _get_score(self, alignment):
        pass

    #alignment feature methods
    def _alignment_features(self,left, right):
        '''
        This function takes as input two alignments and produce features of these
        '''
        return alignment_features(left, right)


    def inspect_alignments(self, match_type = 0, start_score = 'max'):
        '''
            match_type is 0 if you want to inspect non-matches
            and 1 if you want to inspect matches
        '''
        alignments = [(value['score'], value['alignments'], key)  for key, value in self.results.items() if value['match'] == match_type]
        sorted_alignments = sorted(alignments, key=lambda tup: tup[0], reverse = True)

        if start_score == 'max':
            for total_score, alignments, key in sorted_alignments:
                for score, left, right in alignments:
                    for i in range(len(left)):
                        print left[i], right[i]

                    print 'alignment_score: ', score

                print 'total_alignments_score: ', total_score
                print 'key: ', key
                print '\n'

                raw_input("Press Enter to continue...")
        else:
            for total_score, alignments, key in sorted_alignments:
                if total_score > start_score:
                    pass
                else:
                    for score, left, right in alignments:
                        for i in range(len(left)):

                            print left[i], right[i]

                        print 'alignment_score: ', score

                    print 'total_alignments_score: ', total_score
                    print 'key: ', key
                    print '\n'


                raw_input("Press Enter to continue...")


    #plotting functions
    def plot_scores(self):

        matchScores = []
        nonMatchScores = []

        for i in self.bills.keys():
            for j in self.bills.keys():

                if self.scores[i,j] == 0:
                    #ignore if score zero because url is broken
                    pass
                elif i < j and self.bills[i]['match'] == self.bills[j]['match']:
                    matchScores.append(min(self.scores[i,j],200))
                else:
                    nonMatchScores.append(min(self.scores[i,j],200))

        bins = np.linspace(min(nonMatchScores + matchScores), max(nonMatchScores + matchScores), 100)
        plt.hist(nonMatchScores, bins, alpha=0.5, label='Non-Matches')
        plt.hist(matchScores, bins, alpha=0.5, label='Matches')
        plt.legend(loc='upper right')
        plt.xlabel('Alignment Score')
        plt.ylabel('Number of Bill Pairs')
        plt.title('Distribution of Alignment Scores')
        plt.show()

        #make boxplot
        data_to_plot = [matchScores, nonMatchScores]
        fig = plt.figure(1, figsize=(9, 6))
        ax = fig.add_subplot(111)
        bp = ax.boxplot(data_to_plot)
        ax.set_xticklabels(['Match Scores', 'Non-Match Scores'])
        fig.show()


    def low_rank_plot(self):

        #convert dictionary to matrix
        matches = [[value for key, value in values['features'].items()] \
            for keys, values in self.results.items() if values['match'] == 1]

        non_matches = [[value for key, value in values['features'].items()] \
            for keys, values in self.results.items() if values['match'] == 0]

        #matches from 0 to match_index
        match_index = len(matches)

        data = np.array(matches + non_matches)

        sklearn_pca = PCA(n_components=2)
        sklearn_transf = sklearn_pca.fit_transform(data)

        #plot data
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.plot(sklearn_transf[:match_index,0],sklearn_transf[:match_index,1], 
                 'o', markersize=7, color='blue', alpha=0.5, label='matches')
        plt.plot(sklearn_transf[match_index:,0], sklearn_transf[match_index:,1], 
                 '^', markersize=7, color='red', alpha=0.5, label='non-matches')
        plt.xlim([-50,1000])
        plt.ylim([-500,500])
        plt.legend(loc='upper left');
        plt.show()


    ############################################################
    ##alignments utils
    def _diff(self, alignment):
        a = alignment[1]
        b = alignment[2]
        length = min(len(alignment[1]), len(alignment[2]))

        diff = []
        for i in range(length):
            if a[i] == b[i] or a[i] == '-' or b[i] == '-':
                continue
            else:
                diff.append((a[i], b[i]))

        return diff


class DocExperiment(Experiment):

    def _prepare_text_left(self, text, state):

        text = clean_text_for_query(text, state)

        return text.split()


    def _prepare_text_right(self, text, state):

        text = clean_text_for_query(text, state)

        return text.split()


    def _get_score(self, alignment):
        return alignment.alignments[0][0]


class SectionExperiment(Experiment):

    def _prepare_text_left(self, text, state):

        text = clean_text_for_alignment(text, state)

        return map(lambda x: x.split(), text)


    def _prepare_text_right(self, text, state):

        text = clean_text_for_alignment(text, state)

        return flatten(map(lambda x: x.split(), text))


    def _get_score(self, alignment):

        scores = [score for score, left, right in alignment.alignments]

        return sum(scores)


class GridSearch():

    def __init__(self, bills, algorithm, match_scores = [2,3,4], mismatch_scores = [-1,-2,-3], 
                gap_scores = [-1,-2,-3], gap_starts = [-2,-3,-4], gap_extends = [-0.5,-1,-1.5]):
        self.bills = bills
        self.algorithm = algorithm
        if bills == {}:
            self.scores = None
        else:
           self.scores = np.zeros((max(self.bills.keys())+1, max(self.bills.keys())+1))
        self.grid = {}
        self.grid_df = None
        self.match_score = match_scores
        self.mismatch_score = mismatch_scores
        self.gap_score = gap_scores
        self.gap_start = gap_starts
        self.gap_extend = gap_extends


    def grid_search(self):
        #only works for doc experiemnt currently
        #determine parameters to do grid search on for given algorithm
        if self.algorithm == LocalAligner:    
            for match_score in self.match_scores:
                for mismatch_score in self.mismatch_scores:
                    for gap_score in self.gap_scores:

                        print 'running LocalAligner model: match_score {0} mismatch_score {1} gap_score {2}'.format(match_score, mismatch_score, gap_score)

                        e = DocExperiment(self.bills, LocalAligner, match_score = match_score, 
                                            mismatch_score = mismatch_score, gap_score = gap_score)

                        self.evaluate_algorithm()

                        self.grid[(match_score, mismatch_score, gap_score)] = {}
                        self.grid[(match_score, mismatch_score, gap_score)]['results'] = e.results
                        self.grid[(match_score, mismatch_score, gap_score)]['scores'] = e.scores

        elif self.algorithm == AffineLocalAligner:
            for match_score in self.match_scores:
                for mismatch_score in self.mismatch_scores:
                    for gap_start in self.gap_starts:
                        for gap_extend in self.gap_extends:

                            print 'running AffineLocalAligner model: match_score {0} mismatch_score {1} \
                                    gap_start {2} gap_extend'.format(match_score, mismatch_score,
                                                                     gap_start, gap_extend)

                            e = DocExperiment(self.bills, LocalAligner, match_score = match_score, 
                                                mismatch_score = mismatch_score, gap_score = gap_score)

                            e.evaluate_algorithm()

                            self.grid[(match_score, mismatch_score, gap_start, gap_extend)] = {}                            
                            self.grid[(match_score, mismatch_score, gap_start, gap_extend)]['results'] = e.results
                            self.grid[(match_score, mismatch_score, gap_start, gap_extend)]['scores'] = e.scores                            

            return self.grid


    def _create_grid_df(self):
        t = []
        for key1, value1 in grid.items():
            for key2, value2 in value1['results'].items():
                t.append(list(key1) + [key2, value2['score'], value2['match']])
    
        self.grid_df = pd.DataFrame(t)

        if self.algorithm == LocalAligner:
            self.grid_df.columns = ['match_score', 'mismatch_score', 'gap_score', 'pair', 'score', 'match']
        elif self.algorithm == AffineLocalAligner:
            self.grid_df.columns = ['match_score', 'mismatch_score', 'gap_start', 'gap_extend', 'pair', 'score', 'match']

    def plot_grid(self):

        self._create_grid_df()

        df = self.grid_df
        #make maximum possible 500
        df.loc[df['score']>500,'score'] = 500

        #match plot
        df_match = df[(df['mismatch_score'] == -2) & (df['gap_score'] == -1)]

        g = sns.FacetGrid(df_match, col="match_score")
        g = g.map(sns.boxplot, "match", "score")
        sns.plt.ylim(0,400)
        sns.plt.show()

        #mismatch plot
        df_mismatch = df[(df['match_score'] == 3) & (df['gap_score'] == -1)]

        g = sns.FacetGrid(df_mismatch, col="mismatch_score")
        g = g.map(sns.boxplot, "match", "score")
        sns.plt.ylim(0,400)
        sns.plt.show()

        #gap plot
        df_gap = df[(df['match_score'] == 3) & (df['mismatch_score'] == -2)]

        g = sns.FacetGrid(df_gap, col="gap_score")
        g = g.map(sns.boxplot, "match", "score")
        sns.plt.ylim(0,400)
        sns.plt.show()

    def save(self, name):
        with open('../data/grid_results/{0}.p'.format(name), 'wb') as fp:
            pickle.dump(self, fp)


############################################################
##helper function
def alignment_features(left, right):
    '''
    This function takes as input two alignments and produce features of these
    '''
    #alignment features
    features = {}
    features['length'] = len(left)
    features['num_gaps_l'] = 0
    features['num_gaps_r'] = 0
    features['num_mismatches'] = 0
    features['num_matches'] = 0
    features['avg_gap_length_l'] = []
    features['avg_gap_length_r'] = []
    features['avg_consec_match_length'] = []

    #helper variables
    prev_gap_l = False
    prev_gap_r = False
    prev_match = False
    for i in range(len(left)):
        # print 'i: ', i
        # print 'features: ', features
        if left[i] == '-':
            features['num_gaps_l'] += 1
            if not prev_gap_l:
                features['avg_gap_length_l'].append(1)
                prev_gap_l = True
            else:
                features['avg_gap_length_l'][-1] += 1
        else:
            prev_gap_l = False
        if right[i] == '-':
            features['num_gaps_r'] += 1
            if not prev_gap_r:
                features['avg_gap_length_r'].append(1)
                prev_gap_r = True
            else:
                features['avg_gap_length_r'][-1] += 1
        else:
            prev_gap_r = False
        if left[i] != '-' and right[i] != '-':
            if left[i] != right[i]:
                features['num_mismatches'] += 1
            elif left[i] == right[i]:
                features['num_matches'] += 1
                if not prev_match:
                    features['avg_consec_match_length'].append(1)
                    prev_match = True
                else:
                    features['avg_consec_match_length'][-1] += 1
        if left[i] != right[i]:
            prev_match = False

    if features['avg_gap_length_l'] != []:
        features['avg_gap_length_l'] = np.mean(features['avg_gap_length_l'])
    else:
        features['avg_gap_length_l'] = 0
    if features['avg_gap_length_r'] != []:
        features['avg_gap_length_r'] = np.mean(features['avg_gap_length_r'])
    else:
        features['avg_gap_length_r'] = 0
    if features['avg_consec_match_length'] != []:
        features['avg_consec_match_length'] = np.mean(features['avg_consec_match_length'])
    else:
        features['avg_consec_match_length'] = 0

    return features

#good test case: results[(21,22)]
def test_alignment_features():

    def true_features(length, num_gaps_l, num_gaps_r, num_mismatches, 
                        num_matches, avg_gap_length_l, avg_gap_length_r,
                        avg_consec_match_length):
        truth = {}
        truth['length'] = length
        truth['num_gaps_l'] = num_gaps_l
        truth['num_gaps_r'] = num_gaps_r
        truth['num_mismatches'] = num_mismatches
        truth['num_matches'] = num_matches
        truth['avg_gap_length_l'] = avg_gap_length_l
        truth['avg_gap_length_r'] = avg_gap_length_r
        truth['avg_consec_match_length'] = avg_consec_match_length

        return truth

    def check_features_truth(truth, features):
        for field in features.keys():
            if features[field] != truth[field]:
                print field + ' is inconsistent'


    a1 = range(1,10) + ['-', '-'] + range(12,15)
    a2 = range(1,4) + ['-', '-', 7] + range(7,15)

    features = alignment_features(a1, a2)
    truth = true_features(14,2,2,1,9,2,2,3)

    print 'first test: '
    check_features_truth(truth, features)

    b1 = range(1,10)
    b2 = ['-','-',3,5,6,7, '-', '-', 9]
    features = alignment_features(b1, b2)
    truth = true_features(9,0,4,3,2,0,2,1)

    print 'second test: '
    check_features_truth(truth, features)



############################################################
##data creating, saving, and loading

def load_bills():
    with open('../data/bills.p','rb') as fp:
        bills =pickle.load(fp)

    return bills

def load_pickle(name):
    with open('../data/{0}.p'.format(name),'rb') as fp:
        f =pickle.load(fp)

    return f

def load_scores():
    scores = np.load('../data/scores.npy')

    return scores


def save_results(results):
    with open('../data/results.json','wb') as fp:
        pickle.dump(results, fp)


def load_results():
    with open('../data/results.json','rb') as fp:
        data =pickle.load(fp)
    return data



if __name__ == '__main__':

    bills = load_bills()

    print 'testing local alignment'
    e = DocExperiment(bills, LocalAligner)

    e.evaluate_algorithm()

    e.save('local_experiment')

    # print 'testing section local alignment'
    # e = SectionExperiment(bills, SectionLocalAlignment)
    # e.evaluate_algorithm()

    # e.save('section_experiment')

    # print 'testing grid alignment'


