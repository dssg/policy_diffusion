import numpy as np
from numba import jit
from alignmentFunctions import seqToAlign

'''
class LocalSequenceAligner(SequenceAligner):

    def __init__(self, scoring, gapScore, minScore=None):
        super(LocalSequenceAligner, self).__init__(scoring, gapScore)
        self.minScore = minScore

    def computeAlignmentMatrix(self, first, second):
        m = len(first) + 1
        n = len(second) + 1
        f = numpy.zeros((m, n), int)
        for i in xrange(1, m):
            for j in xrange(1, n):
                # Match elements.
                ab = f[i - 1, j - 1] \
                    + self.scoring(first[i - 1], second[j - 1])

                # Gap on sequenceA.
                ga = f[i, j - 1] + self.gapScore

                # Gap on sequenceB.
                gb = f[i - 1, j] + self.gapScore

                f[i, j] = max(0, max(ab, max(ga, gb)))
        return f

    def bestScore(self, f):
        return f.max()

    def backtrace(self, first, second, f):
        m, n = f.shape
        alignments = list()
        alignment = self.emptyAlignment(first, second)
        if self.minScore is None:
            minScore = self.bestScore(f)
        else:
            minScore = self.minScore
        for i in xrange(m):
            for j in xrange(n):
                if f[i, j] >= minScore:
                    self.backtraceFrom(first, second, f, i, j,
                                       alignments, alignment)
        return alignments

    def backtraceFrom(self, first, second, f, i, j, alignments, alignment):
        if f[i, j] == 0:
            alignments.append(alignment.reversed())
        else:
            c = f[i, j]
            p = f[i - 1, j - 1]
            x = f[i - 1, j]
            y = f[i, j - 1]
            a = first[i - 1]
            b = second[j - 1]
            if c == p + self.scoring(a, b):
                alignment.push(a, b, c - p)
                self.backtraceFrom(first, second, f, i - 1, j - 1,
                                   alignments, alignment)
                alignment.pop()
            else:
                if c == y + self.gapScore:
                    alignment.push(alignment.gap, b, c - y)
                    self.backtraceFrom(first, second, f, i, j - 1,
                                       alignments, alignment)
                    alignment.pop()
                if c == x + self.gapScore:
                    alignment.push(a, alignment.gap, c - x)
                    self.backtraceFrom(first, second, f, i - 1, j,
                                       alignments, alignment)
                    alignment.pop()
'''

@jit
def computeAlignmentMatrix(left,right,match_score=3,mismatch_score=-1, gap_score=-2):
    m = len(left) + 1
    n = len(right) + 1
    score_matrix = np.zeros((m, n), int)
    pointer_matrix = np.zeros((m,n), int)
    scores = np.zeros((4), int)
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
    s,p=computeAlignmentMatrix(left,right,match_score=3,mismatch_score=-1, gap_score=-2)

    score = s.max()

    l,r = backtrace(left,right,s,p,match_score=3,mismatch_score=-1, gap_score=-2, gap = '-')

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

if __name__ == '__main__':
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


