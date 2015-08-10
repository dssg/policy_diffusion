'''
module that contains Alignment class and sub classes
'''

from __future__ import division
import numpy as np
import sys
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
from utils.utils import find_subsequence
# from alignment.sequence import Sequence
# from alignment.vocabulary import Vocabulary
# from alignment.sequencealigner import SimpleScoring, LocalSequenceAligner

#TODO: use alignment algorithm
#repsrents two aligned pieces of text
class Alignment(object):

    def __init__(self,left_text,right_text,alignments,alignment_indices,
            left_metadata = {},right_metadata={}):
        

        self.left_text = left_text
        self.right_text = right_text

        alignments.sort(key = lambda x:x[0],reverse = True)
        
        self.alignments = alignments
        self.alignment_indices = alignment_indices
        
    
    def __unicode__(self):
        output_string = ""
        u" ".join([output_string,u"alignments:   \n\n"])            
        
        for i,alignment in enumerate(self.alignments):
            line_breaker = u"\n{0}\n\n".format(i)
            
            score = u"{:.2f}".format(alignment[0])
            l = u" ".join(alignment[1])
            l = u"--------------LEFT TEXT--------------\n{0}\n------------------LEFT TEXT------------------".format(l)
            r = u" ".join(alignment[2])
            r = u"--------------RIGHT TEXT--------------\n{0}\n-----------------RIGHT TEXT-----------------".format(r)
            content = u"score: {0}\n\n{1}\n{2}".format(score,l,r)
            output_string = u"{0}{1}{2}".format(output_string,line_breaker,content)
        
        return output_string
    
    def __str__(self):

        return self.__unicode__().encode("utf-8")

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

        score_matrix, pointer_matrix = self._compute_matrix(a_ints, b_ints,self.match_score,
                self.mismatch_score, self.gap_score)

        l, r, score, align_index = self._backtrace(a_ints, b_ints, score_matrix, pointer_matrix)

        # delta = time.time() - t1
        # print 'local backtrace time: ', delta


        reverse_word_map = {v:k for k,v in word_map.items()}
        reverse_word_map["-"] = "-" 
        l = [reverse_word_map[w] for w in l]
        r = [reverse_word_map[w] for w in r]
        
        alignment_indices.append(align_index)
        alignments.append((score, l, r))
        
        return Alignment(left,right,alignments,alignment_indices)

    def align_by_section(self, left_sections, right):
        
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


        # t1 = time.time()

        H, F, E = self._compute_matrix(a_ints, b_ints, 
                                        self.match_score, self.mismatch_score, 
                                        self.gap_start, self.gap_extend)

        # delta = time.time() - t1
        # print 'affine compute_matrix time: ', delta

        t1 = time.time()
        
        l, r, score, align_index = self._backtrace(a_ints, b_ints, H , F, E, 
                                                self.match_score, self.mismatch_score, 
                                                self.gap_start, self.gap_extend)

        # delta = time.time() - t1
        # print 'affine backtrace time: ', delta

        score = H.max()

        reverse_word_map = {v:k for k,v in word_map.items()}
        reverse_word_map["-"] = "-" 
        l = [reverse_word_map[w] for w in l]
        r = [reverse_word_map[w] for w in r]

        alignment_indices.append(align_index)
        alignments.append((score, l, r))
    
        return Alignment(left,right,alignments,alignment_indices)


    def align_by_section(self, left_sections, right):
        
        alignments = []
        alignment_indices = []

        #keeps track of beginning index of section
        #to recover correct indices of sections
        section_start_index = 0

        for left in left_sections:

            a_ints, b_ints, word_map = self._transform_text(left, right)

            H, F, E = self._compute_matrix(a_ints, b_ints, 
                                        self.match_score, self.mismatch_score, 
                                        self.gap_start, self.gap_extend)

            l, r, score, align_index = self._backtrace(a_ints, b_ints, H , F, E, 
                                                    self.match_score, self.mismatch_score, 
                                                    self.gap_start, self.gap_extend)

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


    @jit
    def _compute_matrix(self, left, right, match_score=3, mismatch_score=-1, gap_start=-2, gap_extend = -.5):
        
        m = len(left) + 1
        n = len(right) + 1
        H = np.zeros((m, n), dtype =float) #best match ending at i,j
        F = np.zeros((m, n), dtype =float) #best ending in space in X
        E = np.zeros((m, n), dtype =float) #best ending in space in Y

        gap_score = np.zeros((2), dtype= float)
        H_score = np.zeros((4), dtype= float) 

        for i in range(1,m):

            score = gap_start + i*gap_extend
            H[i,0] = score
            E[i,0] = score
            H[0,i] = score
            F[0,i] = score

        for i in xrange(1, m):
            for j in xrange(1, n):

                gap_score[0] = F[i-1,j]+gap_extend
                gap_score[1] = H[i-1,j]+gap_start+gap_extend
                F[i,j] = np.max(gap_score)

                gap_score[0] = E[i,j-1]+gap_extend
                gap_score[1] = H[i,j-1] + gap_start + gap_extend
                E[i,j] = np.max(gap_score) 


                H_score[1] = F[i,j]
                H_score[2] = E[i,j]
                if left[i-1] == right[j-1]:
                    H_score[3] = H[i-1,j-1]+match_score
                    H[i,j] = np.max(H_score)
                else:
                    H_score[3] = H[i-1,j-1]+mismatch_score
                    H[i,j] = np.max(H_score)
        
        return H, F, E


    @jit
    def _backtrace(self, left, right, H, F, E, match_score,
                 mismatch_score, gap_start, gap_extend):

        i,j = np.unravel_index(H.argmax(), H.shape)

        score = H[i,j]

        #store start of alignment index
        align_index = {}
        align_index['left_end'] = i
        align_index['right_end'] = j

        #to get multiple maxs, just set score_matrix.argmax() to zero and keep applying argmax for as many as you want
        # decision = pointer_matrix[i,j]

        left_alignment = []
        right_alignment = []
        while H[i,j] != 0:

            #determine whether to insert gaps and how many
            #left gaps
            if H[i,j] == E[i,j]:
                while E[i,j] == E[i, j-1] + gap_extend:
                    j -= 1
                    right_alignment = [right[j]] + right_alignment
                    left_alignment = ['-'] + left_alignment        
        
                j -= 1
                right_alignment = [right[j]] + right_alignment
                left_alignment = ['-'] + left_alignment

            #right gaps
            if H[i,j] == F[i,j]:
                while F[i,j] == F[i-1,j] + gap_extend:
                    i -= 1
                    left_alignment = [left[i]] + left_alignment
                    right_alignment = ['-'] + right_alignment

                i -= 1
                left_alignment = [left[i]] + left_alignment
                right_alignment = ['-'] + right_alignment

            #determine whether it is a match or mismatch
            if left[i-1] == right[j-1]:
                insert_score = match_score
            else:
                insert_score = mismatch_score

            #do not insert gap
            if H[i,j] == H[i-1,j-1] + insert_score: 
                i -= 1
                j -= 1
                left_alignment = [left[i]] + left_alignment
                right_alignment = [right[j]] + right_alignment                                                           

        align_index['left_start'] = i
        align_index['right_start'] = j

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
def create_doc_test_cases():
    #tests
    t1 = ['a']*100
    t2 = ['b']*50 + ['a','a','b']*50

    s1 = [1]*100
    s2 = [2]*50 + [1,1,2]*50

    v1 = np.array([0, 1, 2, 3, 4, 7, 6, 3, 2, 1, 3])
    v2  = np.array([0, 1, 2, 3, 4, 4, 5, 2, 1, 2, 2])

    w1 = np.array([7, 6, 3, 2, 1, 3, 0, 1, 2, 3, 4])
    w2  = np.array([4, 5, 2, 1, 2, 2, 0, 1, 2, 3, 4])

    tests = [(t1,t2), (s1,s2),(v1,v2), (w1,w2), (np.random.choice(5, 30),np.random.choice(5, 30)), \
    (np.array([1, 2, 0, 0, 1, 2, 3, 0, 1, 3, 0, 4, 3, 3, 0, 3, 0, 2, 0, 4, 3, 4, 2, \
       1, 1, 1, 1, 1, 0, 1]), np.array([2, 0, 3, 1, 2, 4, 0, 1, 3, 0, 1, 4, 1, 3, 1, 4, 0, 0, 1, 2, 4, 0, 0, \
       2, 4, 1, 3, 2, 2, 4]))]

    # print tests

    return tests


#LocalAligner algorithm tests
def LocalAligner_unit_tests():
    
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
    tests = create_doc_test_cases()
    for test in tests:
        z1, z2 = test
        test_alignment(z1,z2)

        f = LocalAligner()
        alignment=f.align(z1,z2) #default score is 3,-1,-2

        score, l, r  = alignment.alignments[0]

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


def generic_doc_unit_test(algorithm):

    tests = create_doc_test_cases()
    for test in tests:
        z1, z2 = test
        test_alignment(z1,z2, algorithm)


def LocalAligner_speed_test():
    
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


def generic_doc_speed_test(algorithm):
    '''
    compares speed of algorithm to local alignment algorithm
    '''
    
    input_sizes = [np.exp2(p) for p in range(2,7)]

    average_alg_times = []
    average_local_times = []
    for input_size in input_sizes:
        print input_size
        v1 = np.random.randint(0,10,input_size)
        v2 = np.random.randint(0,10,input_size)
        local_times = []
        alg_times = []
        f = LocalAligner()
        g = algorithm()
        for i in range(2):
            t1 = time.time()
            f.align(v1,v2)
            local_times.append(time.time()-t1)
            
            t2 = time.time()
            g.align(v1,v2)
            alg_times.append(time.time()-t2)
    
        average_local_times.append(np.mean(local_times))
        average_alg_times.append(np.mean(alg_times))
    
    plt.plot(input_sizes,average_local_times, color = 'b', label = 'local alignment')
    plt.plot(input_sizes,average_alg_times, color='r', label = g._algorithm_name)
    plt.legend(loc='upper right')
    plt.xlabel('input size')
    plt.ylim(0,0.02)
    plt.show()


def doc_test_alignment_indices(algorithm):
    #tests
    tests = create_doc_test_cases()

    good_job = True
    for test in tests:
        left_text, right_text = test
        f = algorithm()
        Alignment = f.align(left_text,right_text)
        left, right = clean_alignment(Alignment.alignments[0])

        left_start, left_end = find_subsequence(left, left_text)
        right_start, right_end = find_subsequence(right, right_text)

        if Alignment.alignment_indices[0]['left_start'] != left_start or \
            Alignment.alignment_indices[0]['left_end'] != left_end or \
            Alignment.alignment_indices[0]['right_start'] != right_start or \
            Alignment.alignment_indices[0]['right_end'] != right_end:

            print 'alignment length: ', len(left)

            print 'indices are messed up'

            print 'left_start: ', Alignment.alignment_indices[0]['left_start']
            print 'true left_start: ', left_start
            print 'left_end: ', Alignment.alignment_indices[0]['left_end']
            print 'true left_end', left_end
            print '\n'

            print 'right_start: ', Alignment.alignment_indices[0]['right_start']
            print 'true right_start: ', right_start
            print 'right_end: ', Alignment.alignment_indices[0]['right_end']
            print 'true right_end: ', right_end

            print '\n'

            good_job = False

    if good_job:
        print 'indices worked'


#SectionLocalAlignment Tests
def create_section_tests():
    tests = create_doc_test_cases()

    #convert tests into sections so 
    #that it makes sense for case
    left_test = []
    right_test = []
    for test1, test2 in tests:
        left_test.append(list(test1))
        right_test.append(list(test2))

    return left_test, right_test


def section_unit_tests(Algorithm):
    left_test, right_test = create_section_tests()

    f = Algorithm()
    Alignment = f.align_by_section(left_test, flatten(right_test))

    good_job = True
    for score, left, right in Alignment.alignments:
        true_score = f.alignment_score(left, right)
        if true_score != score:
            print 'left: ', left
            print 'right: ', right
            print 'true alignment score: ', true_score
            print 'calculated score: ', score
            good_job = False

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
            f = LocalAligner()
            f.align_by_section(w1,v2)
            section_times.append(time.time()-t2)
    
        average_local_times.append(np.mean(local_times))
        average_section_times.append(np.mean(section_times))
    
    plt.plot(input_sizes,average_section_times, color = 'b', label = 'section local alignment')
    plt.plot(input_sizes,average_local_times, color='r', label = 'local alignment')
    plt.legend(loc='upper right')
    plt.xlabel('input size')
    plt.ylim(0,0.02)
    plt.show()


def section_test_alignment_indices():
    left_test, right_test = create_section_tests()
    left_test_flattened = flatten(left_test)
    right_test_flattened = flatten(right_test)

    f = LocalAligner()
    Alignment = f.align_by_section(left_test, right_test_flattened)

    good_job = True
    for i in range(len(Alignment.alignments)):
        left, right = clean_alignment(Alignment.alignments[i])

        # print 'left: ', left
        # print 'left_flattened: ', left_test_flattened
        # print '\n'

        # print 'right: ', right
        # print 'right_flattened: ', right_test_flattened
        # print '\n'

        print 'alignment length: ', len(left)


        left_start, left_end = find_subsequence(left, left_test_flattened)
        right_start, right_end = find_subsequence(right, right_test_flattened)

        if Alignment.alignment_indices[i]['left_start'] != left_start or \
            Alignment.alignment_indices[i]['left_end'] != left_end or \
            Alignment.alignment_indices[i]['right_start'] != right_start or \
            Alignment.alignment_indices[i]['right_end'] != right_end:
            
            print 'indices are messed up: '

            print 'left_start: ', Alignment.alignment_indices[i]['left_start']
            print 'true left_start: ', left_start
            print 'left_end: ', Alignment.alignment_indices[i]['left_end']
            print 'true left_end', left_end
            print '\n'

            print 'right_start: ', Alignment.alignment_indices[i]['right_start']
            print 'true right_start: ', right_start
            print 'right_end: ', Alignment.alignment_indices[i]['right_end']
            print 'true right_end: ', right_end

            print '\n'

            good_job = False

    if good_job:
        print 'indices worked'


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
    # print "running LocalAligner unit tests.... \n"
    # LocalAligner_unit_tests()

    # print "running LocalAligner speed tests.... \n"
    # LocalAligner_speed_test()

    # print "running LocalAligner index tests.... \n"
    # doc_test_alignment_indices(LocalAligner)

    print "running AffineLocalAligner unit tests.... \n"
    generic_doc_unit_test(AffineLocalAligner)

    # print "running AffineLocalAligner speed tests.... \n"
    # generic_doc_speed_test(AffineLocalAligner)

    # print "running section unit tests for localaligner.... \n"
    # section_unit_tests(LocalAligner)

    print "running section unit tests for affinealigner.... \n"
    section_unit_tests(AffineLocalAligner)

    # print "running section speed tests.... \n"
    # section_speed_test()

    # print 'running test on keeping track of indices for section algorithm..... \n'
    # section_test_alignment_indices()






