'''
alignment with affine gap
'''

import numpy as np
from numba import jit
# from alignmentFunctions import seqToAlign
from fast_alignment import backtrace


@jit
def computeAlignmentMatrix(left,right,match_score=3,mismatch_score=-1, gap_start=-2, gap_extend = -.5):
    m = len(left) + 1
    n = len(right) + 1
    M = np.zeros((m, n), float) #best ending in match or mismatch
    X = np.zeros((m, n), float) #best ending in space in X
    Y = np.zeros((m, n), float) #
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

#can just feed np.maximum into backtrace algorithm with pointer_matrix?
def affine_backtrace(left, right, M, X, Y, pointer_matrix, gap = '-'):
	left_alignment, right_alignment = backtrace(left, right, np.maximum(M, X, Y), pointer_matrix)
	return left_alignment, right_alignment


def alignment_score(l,r,match_score=3,mismatch_score=-1, gap_start=-2, gap_extend = -.5,  gap = '-'):
    score = 0
    prev_gap = 0 #was previous symbol gap
    for i in range(len(l)):
    	if l[i] == gap or r[i] == gap:
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


def test_alignment(t1,t2):
    M, X, Y,p=computeAlignmentMatrix(t1,t2) #default score is 3,-1,-2

    score = max(M.max(), X.max(), Y.max())

    l,r = affine_backtrace(t1,t2,M, X, Y,p)

    #find score of recovered alignment
    align_score = alignment_score(l,r)

    print 'dp_alg_score: ' + str(score)
    print 'alignment_score: ' + str(align_score)


if __name__ == '__main__':
    t1 = ['a']*100
    t2 = ['b']*50 + ['a','a','b']*50

    s1 = [1]*100
    s2 = [2]*50 + [1,1,2]*50

    v1 = np.array([0, 1, 2, 3, 4, 7, 6, 3, 2, 1, 3])
    v2  = np.array([0, 1, 2, 3, 4, 4, 5, 2, 1, 2, 2])

    w1 = np.array([7, 6, 3, 2, 1, 3, 0, 1, 2, 3, 4])
    w2  = np.array([4, 5, 2, 1, 2, 2, 0, 1, 2, 3, 4])

    x1 = np.array(range(10) + [0,0,0,0] + range(10))
    x2 = np.array(range(10)  + range(10))

    tests = [(t1,t2), (s1,s2),(v1,v2), (w1,w2), (np.random.choice(5, 30),np.random.choice(5, 30)), (x1, x2)]

    for test in tests:
        z1, z2 = test

        test_alignment(z1,z2)

        M, X, Y, p=computeAlignmentMatrix(z1,z2) #default score is 3,-1,-2

        score = np.maximum(M, X, Y).max()

        l,r = affine_backtrace(z1,z2,M, X, Y,p)

        #find score of recovered alignment
        align_score = alignment_score(l,r)
