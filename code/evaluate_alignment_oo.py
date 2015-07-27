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
import re
import pandas as pd
from sklearn.decomposition import PCA
from fast_alignment import *    
from CleanText import model_clean_text_for_alignment, clean_text_for_alignment



class Experiment():

	#TODO: how to include parameters for the algorithms in here
	#use keyword arguments??
	def __init__(self, bills, algorithm):
		self.bills = bills
		self.algorithm = algorithm
		self.scores = np.zeros((max(self.bills.keys())+1, max(self.bills.keys())+1))
		self.results = {}


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

	def evaluate(self):
	    self.evaluate_algorithm(bills)

	    self.plot_scores(scores, bills) 

	    return self.scores, self.results


    @abc.abstractmethod
    def evaluate_algorithm(self):
    	pass


    @abc.abstractmethod
    def clean_bills(self):
    	pass

    #alignment feature methods
    def alignment_features(self,left, right):
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


    @abc.abstractmethod
    def calc_pop_results(self, results):
        pass


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

    def evaluate_algorithm(self, bills, match_score = 3, mismatch_score = -1, gap_score = -2):
        '''
        args:
            matches: dictionary with field corresponding to text and match groups

        returns:
            matrix with scores between all pairs and a dictionary with information
        '''
        for i in self.bills.keys():
            for j in self.bills.keys():
                if i < j: #local alignment gives symmetric distance on document comparison
                    
                    if bills[i] == {} or bills[j] == {}:
                        continue

                    text1 = bills[i]['text'].split()
                    text2 = bills[j]['text'].split()

                    if text1 == '' or text2 == '':
                        continue

                    # Create sequences to be aligned.
                    f = self.algorithm(text1, text2)
                    f.align(match_score, mismatch_score, gap_score)

                    scores[i,j] = alignments[0][0]

                    results[(i,j)] ={}
                    results[(i,j)]['alignments'] = f.alignments
                    results[(i,j)]['score'] = f.alignments[0][0]
                    results[(i,j)]['match'] = (bills[i]['match'] == bills[j]['match'])

                    results[(i,j)]['diff'] = [diff(alignment) for alignment in f.alignments]

                    print 'i: ' + str(i) + ', j: ' + str(j) + ' score: ' + str(alignments[0][0])

        return scores, results



class SectionExperiment(Experiment):

    def evaluate_algorithm(self, bills, match_score = 3, mismatch_score = -1, gap_score = -2):
    '''
    args:
        matches: dictionary with field corresponding to text and match groups

    returns:
        matrix with scores between all pairs and a dictionary with information
    '''
    for i in self.bills.keys():
        for j in self.bills.keys():                

            if bills[i] == {} or bills[j] == {}:
                continue

            text1 = bills[i]['text']
            text2 = bills[j]['text']

            if text1 == '' or text2 == '':
                continue

            # Create sequences to be aligned.
            f = self.algorithm(text1, text2)
            f.align(match_score, mismatch_score, gap_score)

            scores[i,j] = alignments[0][0]

            results[(i,j)] ={}
            results[(i,j)]['alignments'] = f.alignments
            results[(i,j)]['score'] = f.alignments[0][0]
            results[(i,j)]['match'] = (bills[i]['match'] == bills[j]['match'])

            results[(i,j)]['diff'] = [diff(alignment) for alignment in f.alignments]

            print 'i: ' + str(i) + ', j: ' + str(j) + ' score: ' + str(alignments[0][0])

    return scores, results



