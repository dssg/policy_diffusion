from text_alignment import *
from gensim.models import Word2Vec
from evaluation.score_alignments import load_word2vec
from scipy.spatial.distance import cosine

class Word2VecLocalAligner(LocalAligner):

    def __init__(self,match_score = 3, mismatch_score = -1, gap_score = -2):
        LocalAligner.__init__(self, match_score, mismatch_score, gap_score)
        self.model = load_word2vec()
        self._algorithm_name = 'word2vec_local_alignment'

    def __str__(self):
        
        name_str = "{0} instance".format(self._algorithm_name)
        param_str_1 = "match_score = {0}".format(self.gap_score)
        param_str_2 = "mismatch_score = {0}".format(self.match_score)
        param_str_3 = "gap_score = {0}".format(self.mismatch_score)
        return "{0}: {1}, {2}, {3}".format(name_str,param_str_1,param_str_2,param_str_3)


    def align(self,left_sections,right_sections):
        '''
        description:
            find alignments between two documents using word2vec
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

                score_matrix, pointer_matrix = self._compute_matrix(a_ints, b_ints,self.match_score,
                        self.mismatch_score, self.gap_score, self.model)

                l, r, score, align_index = self._backtrace(a_ints, b_ints, score_matrix, pointer_matrix)

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
    def _compute_matrix(self, left, right, match_score, mismatch_score, gap_score, model):
        '''
        description:
            create matrix of optimal scores 
        args:
            left: an array of integers
            right: an array of integers
            match_score: score for match in alignment
            mismatch_score: score for mismatch in alignment
            gap_start: score for first gap 
            gap_extend: score for every gap
            model: word2vec model
        returns:
            three matrices required to construct optimal solution
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
                    scores[1] = score_matrix[i-1,j-1] + mismatch_score*cosine(left[i-1], right[j-1])

                scores[2] = score_matrix[i, j - 1] + gap_score
                
                scores[3] = score_matrix[i - 1, j] + gap_score
        
                max_decision = np.argmax(scores)

                pointer_matrix[i,j] = max_decision
                score_matrix[i,j] = scores[max_decision]
        
        return score_matrix, pointer_matrix