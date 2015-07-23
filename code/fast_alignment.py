'''
module that contains Alignment class and sub classes
'''

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
from alignment.sequence import Sequence
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner, LocalSequenceAligner


class Alignment():

    def __init__(self, left_text, right_text):
        self.left_text = left_text
        self.right_text = right_text
        self.alignments = []
        self.alignment_indices = []

    def _transform_text(self,a,b):
        """Converts a list of words into an numpy array of integers used by the alignment algorithm
        Keyword arguments:
        a -- array of strings
        b -- array of strings
        """
        
        word_map = dict()
        id = 0
        for w in itertools.chain(a,b):
            if w in word_map:
                continue
            else:
                word_map[w] = id
                id +=1
        
        a_ints = np.array([word_map[w] for w in a],dtype = int)
        b_ints = np.array([word_map[w] for w in b],dtype = int)
        return a_ints, b_ints, word_map

    @abc.abstractmethod
    def align():
        pass

class LocalAlignment(Alignment):

    def align(self, match_score=3, mismatch_score=-1, gap_score=-2):
        left = self.left_text
        right = self.right_text

        a_ints, b_ints, word_map = self._transform_text(left, right)
        score_matrix, pointer_matrix = self._compute_matrix(a_ints, b_ints, match_score, mismatch_score, gap_score)
        l, r = self._backtrace(a_ints, b_ints, score_matrix, pointer_matrix)

        score = score_matrix.max()

        reverse_word_map = {v:k for k,v in word_map.items()}
        reverse_word_map["-"] = "-" 
        l = [reverse_word_map[w] for w in l]
        r = [reverse_word_map[w] for w in r]

        return [(score, l,r)]

    @jit
    def _compute_matrix(self, left, right, match_score, mismatch_score, gap_score):

        m = len(left) + 1
        n = len(right) + 1
        score_matrix = np.zeros((m, n),dtype =  float)
        scores = np.zeros((4),dtype = float)
        pointer_matrix = np.zeros((m,n),dtype = int)
        for i in xrange(1, m):
            for j in xrange(1, n):
                
                if left[i-1] == right[j-1]:
                    scores[1] = score_matrix[i-1,j-1] + match_score
                else:
                    scores[1] = score_matrix[i-1,j-1] + mismatch_score

                scores[2] = score_matrix[i, j - 1] + gap_score
                
                scores[3] = score_matrix[i - 1, j] + gap_score
        
                max_decision = np.argmax(scores)

                pointer_matrix[i,j] = max_decision
                score_matrix[i,j] = scores[max_decision]
        
        return score_matrix, pointer_matrix

    @jit
    def _backtrace(self, left, right, score_matrix, pointer_matrix):

        i,j = np.unravel_index(score_matrix.argmax(), score_matrix.shape)

        score = score_matrix[i,j]

        #store start of alignment index
        align_index = {}
        self.alignment_indices.append(align_index)
        self.alignment_indices[-1]['end_1'] = i
        self.alignment_indices[-1]['end_2'] = j

        #to get multiple maxs, just set score_matrix.argmax() to zero and keep applying argmax for as many as you want
        decision = pointer_matrix[i,j]

        left_alignment = []
        right_alignment = []
        while decision != 0 and i > 0 and j > 0:
            if decision == 1: #do not insert space
                i -= 1
                j -= 1
                left_alignment = [left[i]] + left_alignment
                right_alignment = [right[j]] + right_alignment
            elif decision == 2: #insert space in right text
                j -= 1
                right_alignment = [right[j]] + right_alignment
                left_alignment = ['-'] + left_alignment
            elif decision == 3: #insert space in left text
                i -= 1
                left_alignment = [left[i]] + left_alignment
                right_alignment = ['-'] + right_alignment

            #update decision
            decision = pointer_matrix[i,j]

        self.alignment_indices[-1]['start_1'] = i
        self.alignment_indices[-1]['start_2'] = j

        self.alignments.append((score, left_alignment, right_alignment))

        return left_alignment, right_alignment

    def alignment_score(self, l, r,match_score=3, mismatch_score=-1, gap_score=-2):
        score = 0
        for i in range(len(l)):
            if l[i] == r[i]:
                score += match_score
            elif l[i] == '-' or r[i] == '-':
                score += gap_score
            else:
                score += mismatch_score

        return score

###Note: still slow
class AffineLocalAlignment(Alignment):

    # @jit
    def align(self, match_score=3, mismatch_score=-1, gap_start=-2, gap_extend = -.5):
        left = self.left_text
        right = self.right_text

        a_ints, b_ints, word_map = self._transform_text(left, right)
        M, X, Y, pointer_matrix = self._compute_matrix(a_ints, b_ints, match_score, mismatch_score, gap_start, gap_extend)
        l, r = self._backtrace(a_ints, b_ints, M, X, Y, pointer_matrix)

        score = np.maximum(M,X,Y).max()

        reverse_word_map = {v:k for k,v in word_map.items()}
        reverse_word_map["-"] = "-" 
        l = [reverse_word_map[w] for w in l]
        r = [reverse_word_map[w] for w in r]

        return [(score, l,r)]

    @jit
    def _compute_matrix(self, left, right, match_score=3, mismatch_score=-1, gap_start=-2, gap_extend = -.5):
        m = len(left) + 1
        n = len(right) + 1
        M = np.zeros((m, n), float) #best ending in match or mismatch
        X = np.zeros((m, n), float) #best ending in space in X
        Y = np.zeros((m, n), float) #best ending in space in Y
        pointer_matrix = np.zeros((m,n), int)

        #initialize values of matrices
        for i in range(m):
            M[i,0] = np.NINF
        for i in range(n):
            M[0,i] = np.NINF

        for i in range(m):
            X[i,0] = np.NINF
        for i in range(n):
            X[0,i] = gap_start + i * gap_extend

        for i in range(m):
            X[i,0] = gap_start + i * gap_extend
        for i in range(n):
            X[0,i] = np.NINF

        scores = np.zeros((4), int)
        for i in xrange(1, m):
            for j in xrange(1, n):
                
                #0 represents restart below
                if left[i-1] == right[j-1]:
                    M[i,j] = max(max(M[i-1,j-1], X[i-1,j-1], Y[i-1,j-1]) + match_score, 0)
                else:
                    M[i,j] = max(max(M[i-1,j-1], X[i-1,j-1], Y[i-1,j-1]) + mismatch_score, 0)

                X[i,j] = max(gap_start+gap_extend+M[i,j-1], gap_extend + X[i,j-1], gap_start + gap_extend + Y[i,j-1],0)
                
                Y[i,j] = max(gap_start + gap_extend + M[i-1,j], gap_start+gap_extend+X[i-1,j], gap_extend+Y[i-1,j],0)    

                scores = np.array([0, M[i,j], X[i,j], Y[i,j]]) #TODO: does it make sense to terminate backtracing when 0?
                max_decision = np.argmax(scores)

                pointer_matrix[i,j] = max_decision
        
        return M, X, Y, pointer_matrix

    @jit
    def _backtrace(self, left, right, M, X, Y, pointer_matrix):
        
        #only need to know the position with max value
        score_matrix = np.maximum(M, X, Y) 

        i,j = np.unravel_index(score_matrix.argmax(), score_matrix.shape)
        #to get multiple maxs, just set score_matrix.argmax() to zero and keep applying argmax for as many as you want
        decision = pointer_matrix[i,j]

        left_alignment = []
        right_alignment = []
        while decision != 0 and i > 0 and j > 0:
            if decision == 1: #do not insert space
                i -= 1
                j -= 1
                left_alignment = [left[i]] + left_alignment
                right_alignment = [right[j]] + right_alignment
            elif decision == 2: #insert space in right text
                j -= 1
                right_alignment = [right[j]] + right_alignment
                left_alignment = ['-'] + left_alignment
            elif decision == 3: #insert space in left text
                i -= 1
                left_alignment = [left[i]] + left_alignment
                right_alignment = ['-'] + right_alignment

            #update decision
            decision = pointer_matrix[i,j]

        return left_alignment, right_alignment

    def alignment_score(self,l,r,match_score=3,mismatch_score=-1, gap_start=-2, gap_extend = -.5):
        score = 0
        prev_gap = 0 #was previous symbol gap
        for i in range(len(l)):
            if l[i] == '-' or r[i] == '-':
                if prev_gap == 0:
                    prev_gap = 1
                    score += gap_start + gap_extend
                else:
                    score += gap_extend
            elif l[i] == r[i]:
                score += match_score
                prev_gap = 0
            else:
                score += mismatch_score
                prev_gap = 0

        return score


############################################################
##testing functions

#function from python package for testing results
def seqToAlign(a, b, matchScore = 3, mismatchScore = -1, gapScore = -2):
    '''
    args:
        a: list of words
        b: list of words
        matchScore: num
        mismatchScore: num
        gapScore: num
    Returns:
        o/w returns list of tuples with score and top alignments
    Description:
        helper function for finding alignments given a list of words
    '''
    # Create a vocabulary and encode the sequences.
    seq1 = Sequence(a)
    seq2 = Sequence(b)
    v = Vocabulary()
    aEncoded = v.encodeSequence(seq1)
    bEncoded = v.encodeSequence(seq2)

    # Create a scoring and align the sequences using local aligner.
    scoring = SimpleScoring(matchScore, mismatchScore)
    aligner = LocalSequenceAligner(scoring, gapScore)
    score, encodeds = aligner.align(aEncoded, bEncoded, backtrace=True)
    alignments = [v.decodeSequenceAlignment(encoded) for encoded in encodeds]

    return [(a.score, list(a.first), list(a.second)) for a in alignments]


def unit_tests():
    
    def test_alignment(t1,t2):
        f = LocalAlignment(t1,t2)
        alignments=f.align() #default score is 3,-1,-2
        score, l, r  = alignments[0]

        #find score of recovered alignment
        align_score = f.alignment_score(l,r)

        #run package algorithm
        alignments = seqToAlign(t1,t2) #default score is 3,-1,-2

        if score == align_score and score == alignments[0][0]:
            print 'package, backtraced alignment, and alignmnet matrix consistent'
        else:        
            print 'dp_alg_score: ' + str(score)
            print 'alignment_score: ' + str(align_score)
            print 'package_score: ' + str(alignments[0][0])

    #tests
    t1 = ['a']*100
    t2 = ['b']*50 + ['a','a','b']*50

    s1 = [1]*100
    s2 = [2]*50 + [1,1,2]*50

    v1 = np.array([0, 1, 2, 3, 4, 7, 6, 3, 2, 1, 3])
    v2  = np.array([0, 1, 2, 3, 4, 4, 5, 2, 1, 2, 2])

    w1 = np.array([7, 6, 3, 2, 1, 3, 0, 1, 2, 3, 4])
    w2  = np.array([4, 5, 2, 1, 2, 2, 0, 1, 2, 3, 4])

    tests = [(t1,t2), (s1,s2),(v1,v2), (w1,w2), (np.random.choice(5, 30),np.random.choice(5, 30))]
    for test in tests:
        z1, z2 = test
        test_alignment(z1,z2)

        f = LocalAlignment(z1,z2)
        alignments=f.align() #default score is 3,-1,-2

        score, l, r  = alignments[0]

        #run package algorithm
        alignments = seqToAlign(z1,z2) #default score is 3,-1,-2

        l_true, r_true = alignments[0][1:]

        # print 'left_alignment: ', l
        # print 'left_true: ', l_true
        # print 'right_alignment: ', r
        # print 'right_true: ', r_true

        for i in range(len(l)):
            if l[i] != l_true[i]:
                print 'not same sequence'
                break

        for i in range(len(r)):
            if r[i] != r_true[i]:
                print 'not same sequence'
                break


def speed_test():

    v1 = np.random.randint(0,10,200)
    v2 = np.random.randint(0,10,200)
    
    input_sizes = [np.exp2(p) for p in range(2,7)]

    average_our_times = []
    average_package_times = []
    for input_size in input_sizes:
        print input_size
        v1 = np.random.randint(0,10,input_size)
        v2 = np.random.randint(0,10,input_size)
        our_times = []
        package_times = []
        for i in range(2):
            t1 = time.time()
            f = LocalAlignment(v1,v2)
            f.align()
            our_times.append(time.time()-t1)
            
            t2 = time.time()
            seqToAlign(v1,v2)
            package_times.append(time.time()-t2)
    
        average_our_times.append(np.mean(our_times))
        average_package_times.append(np.mean(package_times))
    
    plt.plot(input_sizes,average_package_times, color = 'b')
    plt.plot(input_sizes,average_our_times, color='r')
    plt.ylim(0,0.02)
    plt.show()


def load_results():
    with open('../data/results.json','rb') as fp:
        data =pickle.load(fp)
    return data

if __name__ == '__main__':
    print "running unit tests...."
    unit_tests()

    print "running speed test...."
    speed_test()








