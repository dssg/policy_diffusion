##Test Biopython

import urllib2
from tika import parser
	
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist


'''
Test packages for local alignment with stand your ground laws.

'''

matches = 

u1 ='https://docs.legis.wisconsin.gov/2011/related/proposals/ab69'
doc1 = urllib2.urlopen(u1).read()

u2 = 'http://www.schouse.gov/sess116_2005-2006/bills/4301.htm'
doc2 = urllib2.urlopen(u2).read()

u3 = "http://www.alec.org/model-legislation/72-hour-budget-review-act/"
doc3 = urllib2.urlopen(u3).read()

text1 = parser.from_buffer(doc1)['content']

text2 = parser.from_buffer(doc2)['content'] #matches text1

text3 = parser.from_buffer(doc3)['content'] #does not match the other texts


#################################
###bio
##beware: this is slow/makes my terminal crash
from Bio.Alphabet import SingleLetterAlphabet
from Bio.Seq import Seq

matrix = matlist.blosum62
gap_open = -10
gap_extend = -0.5

test1 = "hellothere"
test2 = "whatishello"

alns = pairwise2.align.globalxx(test1, test2)

alns = pairwise2.align.globalxx(text1, text2)

top_aln = alns[0]

###############################
###alignment
from alignment.sequence import Sequence
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner, LocalSequenceAligner

# Create sequences to be aligned.
a = Sequence(text1.split())
b = Sequence(text3.split())

# Create a vocabulary and encode the sequences.
v = Vocabulary()
aEncoded = v.encodeSequence(a)
bEncoded = v.encodeSequence(b)

# Create a scoring and align the sequences using global aligner.
scoring = SimpleScoring(3, -1)
aligner = LocalSequenceAligner(scoring, -2)
score, encodeds = aligner.align(aEncoded, bEncoded, backtrace=True)

# Iterate over optimal alignments and print them.
for encoded in encodeds:
    alignment = v.decodeSequenceAlignment(encoded)
    print alignment
    print 'Alignment score:', alignment.score
    print 'Percent identity:', alignment.percentIdentity()
