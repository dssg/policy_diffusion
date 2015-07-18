'''
alignment object oriented
'''

from __future__ import division
import numpy as np
from numba import jit
from alignmentFunctions import seqToAlign
import itertools
import time
import matplotlib.pyplot as plt
import sys
import abc
import json
# import io
import pickle

class Alignment():

    def __init__(self, left_text, right_text):
        self.left_text = left_text
        self.right_text = right_text
        self.alignments = []
        self.alignment_indices = []

    def _transform_text(self,a,b):
        """Converts a list of words into an numpy array of integers used by the alignment algorithm
        Keyword arguments:
        t1 -- first text array
        t2 -- second text array
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

# @jit
class LocalAlignment(Alignment):

    # @jit
    def align(self, match_score=3, mismatch_score=-1, gap_score=-2):
        left = self.left_text
        right = self.right_text

        a_ints, b_ints, word_map = self._transform_text(left, right)
        score_matrix, pointer_matrix = self._compute_matrix(a_ints, b_ints, match_score, mismatch_score, gap_score)
        l, r = self._backtrace(a_ints, b_ints, score_matrix, pointer_matrix)

        score = score_matrix.max()

        reverse_word_map = {v:k for k,v in word_map.items()}

        # print 'left_true_array: ', a_ints
        # print "word_map: ", word_map
        # print "reverse_word_map: ", reverse_word_map
        # print "left array: ", l
        # print "------------------------------------------------------"


        reverse_word_map["-"] = "-" 
        l = [reverse_word_map[w] for w in l]
        r = [reverse_word_map[w] for w in r]

        return [(score, l,r)]

    @jit
    def _compute_matrix(self, left, right, match_score, mismatch_score, gap_score):

        m = len(left) + 1
        n = len(right) + 1
        score_matrix = np.zeros((m, n),dtype =  int)
        pointer_matrix = np.zeros((m,n),dtype = int)
        scores = np.zeros((4),dtype = int)
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

    def alignment_score(self, l,r,match_score=3,mismatch_score=-1, gap_score=-2):
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
    def align(self, match_score=3,mismatch_score=-1, gap_start=-2, gap_extend = -.5):
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
    
############################################################
##evaluation functions

def evaluate_algorithm(bills, match_score = 3, mismatch_score = -1, gap_score = -2):
    '''
    args:
        matches: dictionary with field corresponding to text and match groups

    returns:
        matrix with scores between all pairs and a dictionary with information
    '''
    f = LocalAlignment()
    scores = np.zeros((max(bills.keys())+1, max(bills.keys())+1))

    results = {}

    for i in bills.keys():
        for j in bills.keys():
            if i < j: #local alignment gives symmetric distance
                if bills[i] == {} or bills[j] == {}:
                    continue

                text1 = bills[i]['text'].split()
                text2 = bills[j]['text'].split()

                if text1 == '' or text2 == '':
                    continue

                # Create sequences to be aligned.
                alignments = f.align(text1, text2, match_score, mismatch_score, gap_score)

                scores[i,j] = alignments[0][0]

                results[(i,j)] ={}
                results[(i,j)]['alignments'] = [(alignment[1], alignment[2]) for alignment in alignments]
                results[(i,j)]['score'] = alignments[0][0]
                results[(i,j)]['match'] = (bills[i] == bills[j])

                results[(i,j)]['diff'] = diff(alignment)

                print 'i: ' + str(i) + ', j: ' + str(j) + ' score: ' + str(alignments[0][0])

    return scores, results


def plot_scores(scores, bills):

    matchScores = []
    nonMatchScores = []

    for i in bills.keys():
        for j in bills.keys():

            if scores[i,j] == 0:
                #ignore if score zero because url is broken
                pass
            elif i < j and bills[i]['match'] == bills[j]['match']:
                matchScores.append(scores[i,j])
            else:
                nonMatchScores.append(scores[i,j])

    val = 0. 
    plt.plot(matchScores, np.zeros_like(matchScores), 'o')
    plt.plot(nonMatchScores, np.zeros_like(nonMatchScores), 'x')
    plt.plot()
    plt.show()

    #make boxplot
    data_to_plot = [matchScores, nonMatchScores]
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(data_to_plot)
    ax.set_xticklabels(['Match Scores', 'Mismatch Scores'])
    fig.show()


def grid_search(bills, match_scores = [3,4], mismatch_scores = [-1,-2], gap_scores = [-1,-2]):
    grid_performance = {}
    for match_score, mismatch_score, gap_score in zip(match_scores, mismatch_scores, gap_scores):

        grid_performance[(match_score, mismatch_score, gap_score)] = {}
        scores, results = evaluate_algorithm(bills, match_score, mismatch_score, gap_score)
        grid_performance[(match_score, mismatch_score, gap_score)]['results'] = results
        grid_performance[(match_score, mismatch_score, gap_score)]['scores'] = scores


def evaluate():
    with open('/Users/jkatzsamuels/Desktop/dssg/sunlight/policy_diffusion/data/eval.json') as file:
        bills = json.load(file)

    bills = {int(k):v for k,v in bills.iteritems()}

    scores, results = evaluate_algorithm(bills)

    plot_scores(scores, bills) 

    return scores, results


def diff(alignment):
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


def cleanAlignment(alignment):
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


############################################################
##Alignment Feature Generation and Plotting

'''
These functions take as input two alignments and produce features of these
'''

def alignment_features(left, right):
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




def calc_pop_results(results):
    for key,value in results.iteritems():
        left, right = value['alignments'][0]
        features = alignment_features(left, right)
        results[key]['features'] = features

    return results


def plot_alignment_stats(results):
    matches = [value['features'] for key, value in results.items() if value['match'] == 1]
    nonMatches = [value['features'] for key, value in results.items() if value['match'] == 0]
    
    fields = matches[0].keys()
    for field in fields:
        matches_field = [item[field] for item in matches]
        nonMatches_field = [item[field] for item in nonMatches]

        plt.hist(matches_field, 100, alpha=0.5, label='matches')
        plt.hist(nonMatches_field, 100, alpha=0.5, label='non-matches')
        plt.legend(loc='upper right')
        plt.title(field)
        plt.show()

#data saving and loading
def save_results(results):
    with open('../data/results.json','wb') as fp:
        pickle.dump(results, fp)

    # r = [{'key':k, 'value': v} for k, v in results.items()]
    # with io.open('../data/results.json','w', encoding='utf8') as outfile:
    #     json.dumps(r, outfile, ensure_ascii=False)

def load_results():
    with open('../data/results.json','rb') as fp:
        data =pickle.load(fp)
    return data


if __name__ == '__main__':
    print "running unit tests...."
    unit_tests()

    print "running speed test...."
    speed_test()


# import json
# with open('data.txt', 'w') as outfile:
#     json.dump(results, outfile)
