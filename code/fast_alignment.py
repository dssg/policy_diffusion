import numpy as np
from numba import jit

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
def computeAlignmentMatrix(left, right,match_score,gap_score,mismatch_score):
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
    
    return score_matrix,pointer_matrix

