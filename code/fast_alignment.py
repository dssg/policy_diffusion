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
import random
from compiler.ast import flatten
from sklearn.decomposition import PCA
# from alignment.sequence import Sequence
# from alignment.vocabulary import Vocabulary
# from alignment.sequencealigner import SimpleScoring, LocalSequenceAligner

#TODO: use alignment algorithm
#repsrents two aligned pieces of text
class Alignment(object):

    def __init__(self,left_text,right_text,alignments,alignment_indices):
        self.left_text = left_text
        self.right_text = right_text
        self.alignments = alignments
        self.alignment_indices = alignment_indices
    
    def dump_alignment_to_json(self):
        pass

    def annotate_alignment(self):
        pass
    

class Aligner(object):

    def __init__(self):
        if not self._algorithm_name:
            self._algorithm_name = "aligner"


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

class LocalAligner(Aligner):

    def __init__(self,match_score = 3,mismatch_score = -1,gap_score = -2):
        
        self._algorithm_name = "local_alignment"
        
        super(LocalAligner,self).__init__()
        
        self.match_score = match_score
        self.mismatch_score = mismatch_score
        self.gap_score = gap_score


    def __str__(self):
        
        name_str = "{0} instance".format(self._algorithm_name)
        param_str_1 = "match_score = {0}".format(self.gap_score)
        param_str_2 = "mismatch_score = {0}".format(self.match_score)
        param_str_3 = "gap_score = {0}".format(self.mismatch_score)
        return "{0}: {1}, {2}, {3}".format(name_str,param_str_1,param_str_2,param_str_3)

    def align(self,left,right):
        
        
        alignments = []
        alignment_indices = []

        a_ints, b_ints, word_map = self._transform_text(left, right)
        score_matrix, pointer_matrix = self._compute_matrix(a_ints, b_ints, self.match_score,
                self.mismatch_score, self.gap_score)
        l, r, score, align_index = self._backtrace(a_ints, b_ints, score_matrix, pointer_matrix)

        reverse_word_map = {v:k for k,v in word_map.items()}
        reverse_word_map["-"] = "-" 
        l = [reverse_word_map[w] for w in l]
        r = [reverse_word_map[w] for w in r]

        alignment_indices.append(align_index)
        alignments.append((score, l, r))
    
        return Alignment(left,right,alignments,alignment_indices)

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
        align_index['left_end'] = i
        align_index['right_end'] = j

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

        align_index['left_start'] = i
        align_index['right_start'] = j

        # self.alignments.append((score, left_alignment, right_alignment))

        return left_alignment, right_alignment, score, align_index

    def alignment_score(self, l, r):
        score = 0
        for i in range(len(l)):
            if l[i] == r[i]:
                score += self.match_score
            elif l[i] == '-' or r[i] == '-':
                score += self.gap_score
            else:
                score += self.mismatch_score

        return score

###Note: still slow
class AffineLocalAligner(LocalAligner):

    def __init__(self, match_score=3, mismatch_score=-1, gap_start=-3, gap_extend = -.5):
        
        self._algorithm_name = "affine_local_alignment"
        
        super(AffineLocalAligner,self).__init__()
        
        self.match_score = match_score
        self.mismatch_score = mismatch_score
        self.gap_start = gap_start
        self.gap_extend = gap_extend


    def align(self, left, right):

        alignments = []
        alignment_indices = []

        a_ints, b_ints, word_map = self._transform_text(left, right)
        H, pointer_matrix = self._compute_matrix(a_ints, b_ints, 
                                        self.match_score, self.mismatch_score, 
                                        self.gap_start, self.gap_extend)
        
        l, r, score, align_index = self._backtrace(a_ints, b_ints, H , pointer_matrix, self.match_score, self.mismatch_score, 
                                        self.gap_start, self.gap_extend)

        score = H.max()

        reverse_word_map = {v:k for k,v in word_map.items()}
        reverse_word_map["-"] = "-" 
        l = [reverse_word_map[w] for w in l]
        r = [reverse_word_map[w] for w in r]

        alignment_indices.append(align_index)
        alignments.append((score, l, r))
    
        return Alignment(left,right,alignments,alignment_indices)


    @jit
    def _compute_matrix(self, left, right, match_score=3, mismatch_score=-1, gap_start=-2, gap_extend = -.5):
        m = len(left) + 1
        n = len(right) + 1
        H = np.zeros((m, n), float) #best match ending at i,j
        F = np.zeros((m, n), float) #best ending in space in X
        E = np.zeros((m, n), float) #best ending in space in Y
        pointer_matrix = np.zeros((m,n), int)

        H[0,0] = 0
        for i in range(1,m):
            score = gap_start + i*gap_extend
            H[i,0] = score
            E[i,0] = score
            H[0,i] = score
            F[0,i] = score

        for i in xrange(1, m):
            for j in xrange(1, n):

                F[i,j] = max(F[i-1,j]+gap_extend, H[i-1,j]+gap_start+gap_extend)

                E[i,j] = max(E[i,j-1]+gap_extend, H[i,j-1] + gap_start + gap_extend)

                if left[i-1] == right[j-1]:
                    scores = np.array([0,  H[i-1,j-1]+match_score, F[i,j], E[i,j]])
                    H[i,j] = max(scores)
                    max_decision = np.argmax(scores)
                else:
                    scores = np.array([0,  H[i-1,j-1]+mismatch_score, F[i,j], E[i,j]])
                    H[i,j] = max(scores)
                    max_decision = np.argmax(scores)

                pointer_matrix[i,j] = max_decision        
        
        return H, pointer_matrix

    # @jit
    # def _compute_matrix(self, left, right, match_score=3, mismatch_score=-1, gap_start=-2, gap_extend = -.5):
    #     m = len(left) + 1
    #     n = len(right) + 1
    #     M = np.zeros((m, n), float) #best ending in match or mismatch
    #     X = np.zeros((m, n), float) #best ending in space in X
    #     Y = np.zeros((m, n), float) #best ending in space in Y
    #     pointer_matrix = np.zeros((m,n), int)

    #     #initialize values of matrices
    #     for i in range(m):
    #         M[i,0] = np.NINF
    #     for i in range(n):
    #         M[0,i] = np.NINF

    #     for i in range(m):
    #         X[i,0] = np.NINF
    #     for i in range(n):
    #         X[0,i] = gap_start + i * gap_extend

    #     for i in range(m):
    #         X[i,0] = gap_start + i * gap_extend
    #     for i in range(n):
    #         X[0,i] = np.NINF

    #     scores = np.zeros((4), int)
    #     for i in xrange(1, m):
    #         for j in xrange(1, n):
                
    #             #0 represents restart below
    #             if left[i-1] == right[j-1]:
    #                 M[i,j] = max(max(M[i-1,j-1], X[i-1,j-1], Y[i-1,j-1]) + match_score, 0)
    #             else:
    #                 M[i,j] = max(max(M[i-1,j-1], X[i-1,j-1], Y[i-1,j-1]) + mismatch_score, 0)

    #             X[i,j] = max(gap_start+gap_extend+M[i,j-1], gap_extend + X[i,j-1], gap_start + gap_extend + Y[i,j-1],0)
                
    #             Y[i,j] = max(gap_start + gap_extend + M[i-1,j], gap_start+gap_extend+X[i-1,j], gap_extend+Y[i-1,j],0)    

    #             scores = np.array([0, M[i,j], X[i,j], Y[i,j]]) #TODO: does it make sense to terminate backtracing when 0?
    #             max_decision = np.argmax(scores)

    #             pointer_matrix[i,j] = max_decision
        
    #     return M, X, Y, pointer_matrix

    @jit
    def _backtrace(self, left, right, H, pointer_matrix, match_score,
                 mismatch_score, gap_start, gap_extend):

        i,j = np.unravel_index(H.argmax(), H.shape)

        score = H[i,j]

        #store start of alignment index
        align_index = {}
        align_index['left_end'] = i
        align_index['right_end'] = j

        #to get multiple maxs, just set score_matrix.argmax() to zero and keep applying argmax for as many as you want
        decision = pointer_matrix[i,j]

        left_alignment = []
        right_alignment = []
        while H[i,j] > 0 and i > 0 and j > 0:
            if left[i-1] == right[j-1]:
                score = match_score
            else:
                score = mismatch_score
            if H[i,j] == H[i-1,j-1] + score: #do not insert space
                i -= 1
                j -= 1
                left_alignment = [left[i]] + left_alignment
                right_alignment = [right[j]] + right_alignment
            elif H[i,j] == H[i,j-1] + gap_start + gap_extend: #insert space in left text
                j -= 1
                right_alignment = [right[j]] + right_alignment
                left_alignment = ['-'] + left_alignment
            elif H[i,j] == H[i-1,j] + gap_start + gap_extend: #insert space in left text
                i -= 1
                left_alignment = [left[i]] + left_alignment
                right_alignment = ['-'] + right_alignment

            #update decision
            decision = pointer_matrix[i,j]

        align_index['left_start'] = i
        align_index['right_start'] = j

        # self.alignments.append((score, left_alignment, right_alignment))

        return left_alignment, right_alignment, score, align_index

    def alignment_score(self,l,r):
        score = 0
        prev_gap = 0 #was previous symbol gap
        for i in range(len(l)):
            if l[i] == '-' or r[i] == '-':
                if prev_gap == 0:
                    prev_gap = 1
                    score += self.gap_start + self.gap_extend
                else:
                    score += self.gap_extend
            elif l[i] == r[i]:
                score += self.match_score
                prev_gap = 0
            else:
                score += self.mismatch_score
                prev_gap = 0

        return score

#aligns left and right text where left text is a list of sections
class SectionLocalAligner(LocalAligner):

    '''
    Input:
        List of lists where each list is a list of words in a section in the document 

    Description:
        Local Alignment that looks for alignments by section
    '''
    def align(self,left_sections,right):
        
        alignments = []
        alignment_indices = []

        #keeps track of beginning index of section
        #to recover correct indices of sections
        section_start_index = 0

        for left in left_sections:

            a_ints, b_ints, word_map = self._transform_text(left, right)
            score_matrix, pointer_matrix = self._compute_matrix(a_ints, b_ints, self.match_score,
                    self.mismatch_score, self.gap_score)
            l, r, score, align_index = self._backtrace(a_ints, b_ints, score_matrix, pointer_matrix)

            score = score_matrix.max()

            reverse_word_map = {v:k for k,v in word_map.items()}
            reverse_word_map["-"] = "-" 
            l = [reverse_word_map[w] for w in l]
            r = [reverse_word_map[w] for w in r]

            alignments.append((score, l, r))
            
            #adjust for section
            align_index['left_start'] += section_start_index
            align_index['left_end'] += section_start_index
            
            alignment_indices.append(align_index)

            #update section_start_index
            section_start_index += len(left)
        
        left = reduce(lambda x,y:x+y,left_sections)

        return Alignment(left,right,alignments,alignment_indices)



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

#testing functions
def create_test_cases():
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

    return tests


#LocalAligner algorithm tests
def unit_tests():
    
    def test_alignment(t1,t2):
        f = LocalAligner()
        alignment=f.align(t1,t2) #default score is 3,-1,-2
        score, l, r  = alignment.alignments[0]

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
    tests = create_test_cases()
    for test in tests:
        z1, z2 = test
        test_alignment(z1,z2)

        f = LocalAligner()
        alignments=f.align(z1,z2) #default score is 3,-1,-2

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
    
    input_sizes = [np.exp2(p) for p in range(2,7)]

    average_our_times = []
    average_package_times = []
    for input_size in input_sizes:
        print input_size
        v1 = np.random.randint(0,10,input_size)
        v2 = np.random.randint(0,10,input_size)
        our_times = []
        package_times = []
        f = LocalAligner()
        for i in range(2):
            t1 = time.time()
            f.align(v1,v2)
            our_times.append(time.time()-t1)
            
            t2 = time.time()
            seqToAlign(v1,v2)
            package_times.append(time.time()-t2)
    
        average_our_times.append(np.mean(our_times))
        average_package_times.append(np.mean(package_times))
    
    plt.plot(input_sizes,average_package_times, color = 'b', label = 'package')
    plt.plot(input_sizes,average_our_times, color='r', label = 'our implementation')
    plt.legend(loc='upper right')
    plt.xlabel('input size')
    plt.ylim(0,0.02)
    plt.show()


#SectionLocalAlignment Tests
def create_section_tests():
    tests = create_test_cases()

    #convert tests into sections so 
    #that it makes sense for case
    left_test = []
    right_test = []
    for test1, test2 in tests:
        left_test.append(list(test1))
        right_test.append(list(test2))

    return left_test, right_test


def section_unit_tests():
    left_test, right_test = create_section_tests()

    f = SectionLocalAligner()
    f.align(left_test,right_test)

    good_job = True
    for score, left, right in f.alignments:
        true_score = f.alignment_score(left, right)
        if true_score != score:
            print 'true alignment score: ', true_score
            print 'calculated score: ', score
            good_job = False
            break

    if good_job:
        print "calculated alignment scores correctly"


def section_speed_test():

    input_sizes = [np.exp2(p) for p in range(2,9)]

    average_local_times = []
    average_section_times = []
    for input_size in input_sizes:
        print input_size
        v1 = list(np.random.randint(0,10,input_size))
        v2 = list(np.random.randint(0,10,input_size))

        cut1 = random.randint(0,len(v1))
        cut2 = random.randint(cut1,len(v2))
        cut3 = random.randint(cut2,len(v2))
        w1 = [v1[:cut1], v1[cut1:cut2], v1[cut2:cut3], v1[cut3:]]

        local_times = []
        section_times = []
        for i in range(2):
            t1 = time.time()
            f = LocalAligner()
            f.align(v1,v2)
            local_times.append(time.time()-t1)

            t2 = time.time()
            f = SectionLocalAligner()
            f.align(w1,v2)
            section_times.append(time.time()-t2)
    
        average_local_times.append(np.mean(local_times))
        average_section_times.append(np.mean(section_times))
    
    plt.plot(input_sizes,average_section_times, color = 'b', label = 'section local alignment')
    plt.plot(input_sizes,average_local_times, color='r', label = 'local alignment')
    plt.legend(loc='upper right')
    plt.xlabel('input size')
    plt.ylim(0,0.02)
    plt.show()


def test_alignment_indices():
    left_test, right_test = create_section_tests()

    f = SectionLocalAligner()
    f.align(left_test, right_test)

    good_job = True
    for i in range(len(f.alignments)):
        left, right = clean_alignment(f.alignments[i])

        left_start, left_end = find_subsequence(left, left_test)
        right_start, right_end = find_subsequence(right, right_test)

        if f.alignment_indices['left_start'] != left_start or \
            f.alignment_indices['left_end'] != left_end or \
            f.alignment_indices['right_start'] != right_start or \
            f.alignment_indices['right_end'] != right_end:
            
            print 'indices are messed up'
            good_job = False
            break

    if good_job:
        print 'indices worked'


def generic_unit_test(algorithm):

    def test_alignment(t1,t2, algorithm):
        f = algorithm()
        alignment=f.align(t1,t2) #default score is 3,-1,-2
        score, l, r  = alignment.alignments[0]

        #find score of recovered alignment
        align_score = f.alignment_score(l,r)

        if score == align_score:
            print 'backtraced alignment and alignmnet matrix consistent'
        else:
            print 'backtraced alignment and alignmnet matrix not consistent'
            print 'dp_alg_score: ' + str(score)
            print 'alignment_score: ' + str(align_score)
    
            print 'left_alignment: ', l 
            print 'right_alignment: ', r

    tests = create_test_cases()
    for test in tests:
        z1, z2 = test
        test_alignment(z1,z2, algorithm)

############################################################
##helper functions
def clean_alignment(alignment):
    '''
    arg:
        alignment object
    returns:
        2 list of alignment words without the alignment symbol
    '''
    keep1 = []
    keep2 = []
    for item in alignment[1]:
        if item != '-':
            keep1.append(item)

    for item in alignment[2]:
        if item != '-':
            keep2.append(item)

    return (keep1, keep2)



if __name__ == '__main__':
    # print "running unit tests...."
    # unit_tests()



    # print "running section_unit_tests"
    # section_unit_tests()

    # print "running section_speed_test"
    # section_speed_test()

    # print 'running test_alignment_indices'
    # test_alignment_indices()

    print 'Running AffineLocalAligner test....'
    generic_unit_test(AffineLocalAligner)





