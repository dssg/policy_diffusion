from __future__ import division
import numpy as np
from numba import jit
from alignmentFunctions import seqToAlign
import itertools
<<<<<<< Updated upstream
import time
import matplotlib.pyplot as plt
import sys

#converts text to arrays of ints
def convert_text_to_ints(a,b):
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

    return a_ints,b_ints,word_map



@jit
def computeAlignmentMatrix(left,right,match_score=3,mismatch_score=-1, gap_score=-2):
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
def backtrace(left, right, score_matrix, pointer_matrix, gap = '-'):
    '''
    returns
        left_alignment
        right_alignment
    '''
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
            left_alignment = [gap] + left_alignment
        elif decision == 3: #insert space in left text
            i -= 1
            left_alignment = [left[i]] + left_alignment
            right_alignment = [gap] + right_alignment

        #update decision
        decision = pointer_matrix[i,j]

    return left_alignment, right_alignment

# @jit
# def backtrace(left, right, score_matrix, pointer_matrix, gap = 0): #0 represents gap
#     '''
#     returns
#         left_alignment
#         right_alignment
#     '''
#     i,j = np.unravel_index(score_matrix.argmax(), score_matrix.shape)
#     #to get multiple maxs, just set score_matrix.argmax() to zero and keep applying argmax for as many as you want
#     decision = pointer_matrix[i,j]

#     length = max(len(left),len(right))
#     left_alignment = np.zeros(length)
#     right_alignment = np.zeros(length)
#     k = length - 1
#     while decision != 0 and i > 0 and j > 0:
#         if decision == 1: #do not insert space
#             i -= 1
#             j -= 1
#             left_alignment[k] = left[i]
#             right_alignment[k] = right[j]
#         elif decision == 2: #insert space in right text
#             j -= 1
#             right_alignment[k] = right[j]
#             left_alignment[k] = gap
#         elif decision == 3: #insert space in left text
#             i -= 1
#             left_alignment[k] = left[i]
#             right_alignment[k] = gap

#         k -= 1

#         #update decision
#         decision = pointer_matrix[i,j]

#     return left_alignment[k:], right_alignment[k:]

def align(left,right,match_score=3,mismatch_score=-1, gap_score=-2, gap = '-'):
    
    left,right,word_map = convert_text_to_ints(left,right)

    s,p=computeAlignmentMatrix(left,right,match_score,mismatch_score, gap_score)

    score = s.max()

    l,r = backtrace(left,right,s,p, gap = '-')


    reverse_word_map = {v:k for k,v in word_map.items()}
    reverse_word_map["-"] = "-" 
    l = [reverse_word_map[w] for w in l]
    r = [reverse_word_map[w] for w in r]


    return [(score, l,r)]


def alignment_score(l,r,match_score=3,mismatch_score=-1, gap_score=-2, gap = '-'):
    score = 0
    for i in range(len(l)):
        if l[i] == r[i]:
            score += match_score
        elif l[i] == gap or r[i] == gap:
            score += gap_score
        else:
            score += mismatch_score

    return score


def test_alignment(t1,t2):
    s,p=computeAlignmentMatrix(t1,t2) #default score is 3,-1,-2

    score = s.max()

    l,r = backtrace(t1,t2,s,p)

    #find score of recovered alignment
    align_score = alignment_score(l,r)

    #run package algorithm
    alignments = seqToAlign(t1,t2) #default score is 3,-1,-2

    print 'dp_alg_score: ' + str(score)
    print 'alignment_score: ' + str(align_score)
    print 'package_score: ' + str(alignments[0][0])

    # l_true, r_true = alignments[0][1:]

    # if len(l) != len(l_true) or len(r) != len(r_true):
    #     print "Not the same alignment"
    #     print 'length of l: ' + str(len(l))
    #     print 'length of true l: ' + str(len(l_true))
    #     print 'length of r: ' + str(len(r))
    #     print 'length of true r: ' + str(len(r_true))

    # for i in range(len(l)):
    #     if l[i] != l_true[i]:
    #         print "Not the same alignment"
    #         print "left alignment difrence: " + str(l[i]) + ',' + str(l_true[i])
    #         print '\n'
    #         break

    # for i in range(len(r)):
    #     if r[i] != r_true[i]:
    #         print "Not the same alignment"
    #         print "right alignment difrence: " + str(r[i]) + ',' + str(r_true[i])
    #         print 'index of failure: ' + str(i)
    #         break


def unit_tests():
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

        s,p=computeAlignmentMatrix(z1,z2) #default score is 3,-1,-2

        score = s.max()

        l,r = backtrace(z1,z2,s,p)

        #find score of recovered alignment
        align_score = alignment_score(l,r)

        #run package algorithm
        alignments = seqToAlign(z1,z2) #default score is 3,-1,-2

        l_true, r_true = alignments[0][1:]

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
    
    input_sizes = [np.exp2(p) for p in range(2,11)]

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
            align(v1,v2)
            our_times.append(time.time()-t1)
            
            t2 = time.time()
            seqToAlign(v1,v2)
            package_times.append(time.time()-t2)
    
        average_our_times.append(np.mean(our_times))
        average_package_times.append(np.mean(package_times))
    
    plt.plot(input_sizes,average_package_times)
    plt.plot(input_sizes,average_our_times)
    plt.show()
    

if __name__ == '__main__':
    print "running unit tests...."
    unit_tests()

    print "running speed test...."
    speed_test()