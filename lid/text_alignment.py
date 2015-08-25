'''
module that contains Alignment class and sub classes
'''

from __future__ import division
import numpy as np
from numba import jit
import itertools
import abc




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
        self.current = 0
        self.last = len(self.alignments)
        
        
    
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
    

    def __getitem__(self, index):
        d = {"left":self.alignments[index][1],
                    "right":self.alignments[index][2],
                    "score":self.alignments[index][0]}
        d.update(self.alignment_indices[index])
        return d


    def __str__(self):

        return self.__unicode__().encode("utf-8")

    def dump_alignment_to_json(self):
        pass

    def annotate_alignment(self):
        pass


class Aligner(object):
    '''
    Aligner is an abstract class for Aligner algorithms.

    Attributes:
        _transform_text: converts two lists of words into two numpy arrays
        align: abstract method for alignment
    '''

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

########################################################################################################################
########################################################################################################################

class LocalAligner(Aligner):
    '''
    LocalAligner is a class for performing the Smith-Waterman algorithm on two documents.

    Attributes:
        match_score: match score for Smith-Waterman algorithm
        mismatch_score: mismatch score for Smith-Waterman algorithm
        gap_score: gap score for Smith-Waterman algorithm
    '''

    def __init__(self,match_score = 3,mismatch_score = -1,gap_score = -2):
        '''
        inits LocalAligner with match_score, mismatch_score, gap_score
        '''
        
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

    def align(self, left_sections, right_sections):
        '''
        description:
            find alignments between two documents
        args:
            left_sections: a list of lists of words
            right_sections: a list of lists of words (usually just a list of a list of words)

        returns:
            alignment object
        '''
        
        alignments = []
        alignment_indices = []

        #keeps track of beginning index of section
        #to recover correct indices of sections
        section_start_index = 0

        for left in left_sections:
            for right in right_sections:

                a_ints, b_ints, word_map = self._transform_text(left, right)

                score_matrix, pointer_matrix = self._compute_matrix(a_ints, b_ints, self.match_score,
                                                                    self.mismatch_score, self.gap_score)

                l, r, score, align_index = self._backtrace(a_ints, b_ints, score_matrix, pointer_matrix)

                reverse_word_map = {v:k for k,v in word_map.items()}
                reverse_word_map["-"] = "-" 
                l = [reverse_word_map[w] for w in l]
                r = [reverse_word_map[w] for w in r]

                #don't add alignment if it has a score of 0
                if score == 0:
                    continue

                alignments.append((score, l, r))
                
                #adjust for section
                align_index['left_start'] += section_start_index
                align_index['left_end'] += section_start_index
                
                alignment_indices.append(align_index)

                #update section_start_index
                section_start_index += len(left)
            
        left = reduce(lambda x,y:x+y,left_sections)
        right = reduce(lambda x,y:x+y,right_sections)

        return Alignment(left,right,alignments,alignment_indices)

    @jit
    def _compute_matrix(self, left, right, match_score, mismatch_score, gap_score):
        '''
        description:
            create matrix of optimal scores
        args:
            left: an array of integers
            right: an array of integers
            match_score: score for match in alignment
            mismatch_score: score for mismatch in alignment
            gap_score: score for gap in alignment
        returns:
            matrix representing optimal score for subsequences ending at each index
            pointer_matrix for reconstructing a solution
        '''

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

        '''
        description:
            backtrace for recovering solution from dp matrix
        args:
            left: an array of integers
            right: an array of integers
            matrix representing optimal score for subsequences ending at each index
            pointer_matrix for reconstructing a solution
        returns:
            left_alignment: array of integers
            right_alignment: array of integers
            score: score of alignment
            align_index: dictionary with indices of where alignment occurs in left and right
        '''

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
        '''
        description:
            computes the score of an alignment using the scoring
            used for checking algorithm
        args:
            l: list of words
            r: list of words
        returns:
            score: number

        '''
        score = 0
        for i in range(len(l)):
            if l[i] == r[i]:
                score += self.match_score
            elif l[i] == '-' or r[i] == '-':
                score += self.gap_score
            else:
                score += self.mismatch_score

        return score

########################################################################################################################
########################################################################################################################

class AffineLocalAligner(LocalAligner):
    '''
    AffineLocalAligner is a class for performing the Smith-Waterman algorithm 
    with an affine penalty on two documents.

    Attributes:
        match_score: match score for Smith-Waterman algorithm
        mismatch_score: mismatch score for Smith-Waterman algorithm
        gap_start: gap score for initial space
        gap_extend: gap score for every gap
        first_gap_in_sequence_of_gaps = gap_score + gap_extend
        every_subsequent_gap_in_sequence_of_gaps = gap_extend
    '''

    def __init__(self, match_score=3, mismatch_score=-1, gap_start=-3, gap_extend = -.5):
        '''
        inits AffineLocalAligner with match_score, mismatch_score, gap_start, and gap_extend
        '''
        self._algorithm_name = "affine_local_alignment"
        
        super(AffineLocalAligner,self).__init__()
        
        self.match_score = match_score
        self.mismatch_score = mismatch_score
        self.gap_start = gap_start
        self.gap_extend = gap_extend

    def align(self, left_sections, right_sections):
        '''
        description:
            find alignments between two documents according to affine scoring
        args:
            left_sections: a list of lists of words
            right_sections: a list of lists of words (usually just a list of a list of words)

        returns:
            alignment object
        '''
        alignments = []
        alignment_indices = []

        for left in left_sections:
            for right in right_sections:

                a_ints, b_ints, word_map = self._transform_text(left, right)

                H, F, E = self._compute_matrix(a_ints, b_ints, 
                                                self.match_score, self.mismatch_score, 
                                                self.gap_start, self.gap_extend)

                
                l, r, score, align_index = self._backtrace(a_ints, b_ints, H , F, E, 
                                                        self.match_score, self.mismatch_score, 
                                                        self.gap_start, self.gap_extend)


                score = H.max()

                reverse_word_map = {v:k for k,v in word_map.items()}
                reverse_word_map["-"] = "-" 
                l = [reverse_word_map[w] for w in l]
                r = [reverse_word_map[w] for w in r]

                alignment_indices.append(align_index)
                alignments.append((score, l, r))

        left = reduce(lambda x,y:x+y,left_sections)
        right = reduce(lambda x,y:x+y,right_sections)
    
        return Alignment(left,right,alignments,alignment_indices)


    @jit
    def _compute_matrix(self, left, right, match_score=3, mismatch_score=-1, gap_start=-2, gap_extend = -.5):
        '''
        description:
            create matrix of optimal scores according to affine scoring system
        args:
            left: an array of integers
            right: an array of integers
            match_score: score for match in alignment
            mismatch_score: score for mismatch in alignment
            gap_start: score for first gap 
            gap_extend: score for every gap
        returns:
            three matrices required to construct optimal solution
        '''
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

        '''
        description:
            backtrace for recovering solution from dp matrix in affine scoring system
        args:
            left: an array of integers
            right: an array of integers
            H: matrix
            F: matrix
            E: matrix
        returns:
            left_alignment: array of integers
            right_alignment: array of integers
            score: score of alignment
            align_index: dictionary with indices of where alignment occurs in left and right
        '''

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
        '''
        description:
            computes the score of an alignment using the affine scoring
            used for checking algorithm
        args:
            l: list of words
            r: list of words
        returns:
            score: number

        '''
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




