
import random
import matplotlib.pyplot as plt
import time
import numpy as np
from compiler.ast import flatten
from alignment.sequence import Sequence
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, LocalSequenceAligner
from utils.general_utils import find_subsequence
from text_alignment import *


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
    a = a[0]
    b = b[0]
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
    t1 = [['a']*100]
    t2 = [['b']*50 + ['a','a','b']*50]

    s1 = [[1]*100]
    s2 = [[2]*50 + [1,1,2]*50]

    v1 = [np.array([0, 1, 2, 3, 4, 7, 6, 3, 2, 1, 3])]
    v2  = [np.array([0, 1, 2, 3, 4, 4, 5, 2, 1, 2, 2])]

    w1 = [np.array([7, 6, 3, 2, 1, 3, 0, 1, 2, 3, 4])]
    w2  = [np.array([4, 5, 2, 1, 2, 2, 0, 1, 2, 3, 4])]

    tests = [(t1,t2), (s1,s2),(v1,v2), (w1,w2), ([np.random.choice(5, 30)],[np.random.choice(5, 30)]), \
    ([np.array([1, 2, 0, 0, 1, 2, 3, 0, 1, 3, 0, 4, 3, 3, 0, 3, 0, 2, 0, 4, 3, 4, 2, \
       1, 1, 1, 1, 1, 0, 1])], [np.array([2, 0, 3, 1, 2, 4, 0, 1, 3, 0, 1, 4, 1, 3, 1, 4, 0, 0, 1, 2, 4, 0, 0, \
       2, 4, 1, 3, 2, 2, 4])])]

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
        v1 = [np.random.randint(0,10,input_size)]
        v2 = [np.random.randint(0,10,input_size)]
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
        v1 = [np.random.randint(0,10,input_size)]
        v2 = [np.random.randint(0,10,input_size)]
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
    
    return average_local_times, average_alg_times


def doc_test_alignment_indices(algorithm):
    #tests
    tests = create_doc_test_cases()

    good_job = True
    for test in tests:

        left_text, right_text = test
        try:
            left_text[0] = left_text[0].tolist()
            right_text[0] = right_text[0].tolist()
        except:
            pass
        f = algorithm()
        Alignment = f.align(left_text,right_text)
        left, right = clean_alignment(Alignment.alignments[0])


        left_start, left_end = find_subsequence(left, flatten(left_text))
        right_start, right_end = find_subsequence(right, flatten(right_text))

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
        left_test.append(list(test1[0]))
        right_test.append(list(test2[0]))

    return left_test, right_test


def section_unit_tests(Algorithm):
    left_test, right_test = create_section_tests()

    f = Algorithm()
    Alignment = f.align(left_test, [flatten(right_test)])

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
        v1 = [np.random.randint(0,10,input_size)]
        v2 = [np.random.randint(0,10,input_size)]

        cut1 = random.randint(0,len(v1))
        cut2 = random.randint(cut1,len(v2))
        cut3 = random.randint(cut2,len(v2))
        w1 = [v1[0][:cut1], v1[0][cut1:cut2], v1[0][cut2:cut3]]

        local_times = []
        section_times = []
        for i in range(2):
            t1 = time.time()
            f = LocalAligner()
            f.align(v1,v2)
            local_times.append(time.time()-t1)

            t2 = time.time()
            f = LocalAligner()
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


def section_test_alignment_indices():
    left_test, right_test = create_section_tests()
    left_test_flattened = flatten(left_test)
    right_test_flattened = flatten(right_test)

    f = LocalAligner()
    Alignment = f.align(left_test, [right_test_flattened])

    good_job = True
    for i in range(len(Alignment.alignments)):
        left, right = clean_alignment(Alignment.alignments[i])

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
    print "running LocalAligner unit tests.... \n"
    LocalAligner_unit_tests()

    print "running LocalAligner speed tests.... \n"
    LocalAligner_speed_test()

    print "running LocalAligner index tests.... \n"
    doc_test_alignment_indices(LocalAligner)

    print "running AffineLocalAligner unit tests.... \n"
    generic_doc_unit_test(AffineLocalAligner)

    print "running AffineLocalAligner speed tests.... \n"
    generic_doc_speed_test(AffineLocalAligner)

    print "running section unit tests for localaligner.... \n"
    section_unit_tests(LocalAligner)

    print "running section unit tests for affinealigner.... \n"
    section_unit_tests(AffineLocalAligner)

    print "running section speed tests.... \n"
    section_speed_test()

    print 'running test on keeping track of indices for section algorithm..... \n'
    section_test_alignment_indices()

    print 'running speed test on Word2VecLocalAligner.... \n'