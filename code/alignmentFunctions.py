'''
Local Alignment Functions
'''

#alignment package
from alignment.sequence import Sequence
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner, LocalSequenceAligner

#gensim package
from gensim.models import Word2Vec

#nltk
from nltk.tokenize import sent_tokenize

#other
import numpy as np
from collections import defaultdict

import urllib2
from tika import parser

#################
class Alignment:
    def __init__(self, score, alignments):
        self.score = score
        self.alignments = alignments

class Synonyms:
    def __init__(self, threshold, word2vec_model):
        '''
        center: word that is treated as center of cluster of words
        threshold: number for distance of words to be considered synonyms (given by gensim)
        word2vec_model: word2vec model for finding synonyms
        rep_num: representative number of the Synonyms
        words: the words that belong to the synyonym cluster
        '''
        self.threshold = threshold
        self.model = word2vec_model

        self.centerToSyn = defaultdict(list)
        self.synToCenter = {}
        self.centerToRepNum = {}

	def createRepNum(self):
		if self.rep_num == None:
			self.rep_num = np.random.uniform(0,1)
		else:
			print "rep_num already exists."

	def addWord(self, word):
		if not self.centerToSyn:
			self.addCenter(word)
		elif word in self.centerToSyn or word in self.synToCenter:
			pass
		else:
			#word has not been added to synonyms
			#need to see if there is a center that we should add the word to.
			for center in self.centerToSyn.keys():
				if self.model.similarity(word, center) <= self.threshold:
					self.centerToSyn[center].append(word)
					self.synToCenter[word] = center
					break
			if word not in self.synToCenter:
				self.addCenter(word)

	def addCenter(self, word):
		self.centerToSyn[word].append(word) #TODO: maybe don't include center in list
		self.centerToRepNum[word] = np.random.uniform(0,1)
		self.synToCenter[word] = word

	def makeSynonyms(self, words):
		'''
		text is assumed to be split
		'''
		for word in words:
			self.addWord(word)

#################
#Helper Functions

def seqToAlign(a, b, matchScore = 3, mismatchScore = -1, gapScore = -2):
    '''
    args:
        a: list of words
        b: list of words
        matchScore: num
        mismatchScore: num
        gapScore: num
    Returns:
        alignment object
    Description:
        helper function for finding alignments given a list of words
    '''
    # Create a vocabulary and encode the sequences.
    v = Vocabulary()
    aEncoded = v.encodeSequence(a)
    bEncoded = v.encodeSequence(b)

    # Create a scoring and align the sequences using local aligner.
    scoring = SimpleScoring(matchScore, mismatchScore)
    aligner = LocalSequenceAligner(scoring, gapScore)
    score, encodeds = aligner.align(aEncoded, bEncoded, backtrace=True)
    alignments = [v.decodeSequenceAlignment(encoded) for encoded in encodeds]

    return Alignment(score, alignments)

def seqToSeqSyn(words, syn):
    '''
    args:
        words: list of words
        syn: Synonyms object
    Returns:
        a lists of words transformed by inserting synonyms representatives in
    Description:
        Make a sequence of text that does not distinguish between synonyms
    '''
    for i in range(len(words)):
        word = words[i]
        center = syn.synToCenter[word]
        if len(syn.centerToSyn[center]) > 1:
            words[i] = str(syn.centerToRepNum[center])

	return words

#################
#Main Functions

def align(text1, text2, matchScore = 3, mismatchScore = -1, gapScore = -2):
    '''
    args:
        text1: string corresponding to text
        text2: string corresponding to text
        matchScore: num
        mismatchScore: num
        gapScore: num
    Returns:
        alignment object
    Description:
        function for finding local alignments (with no synonym modification)
    '''
    return seqToAlign(text1.split(), text2.split(), matchScore , mismatchScore, gapScore)


def word2vecAlign(text1, text2, word2vec_model,threshold = .05, matchScore = 3, mismatchScore = -1, gapScore = -2):
    '''
    args:
        text1: string corresponding to text
        text2: string corresponding to text
        matchScore: num
        mismatchScore: num
        gapScore: num
        threshold: num for categorizing two words as synonyms
        word2vec_model: model used for considering two words synonyms
    Returns:
        alignment object
    Description:
        function for finding local alignments (with no synonym modification)
    '''
    seq1 = text1.split()
    seq2 = text2.split()

    #make synonym object and create synonyms
    syn = Synonyms(threshold, word2vec_model)

    syn.makeSynonyms(seq1)
    syn.makeSynonyms(seq2)

    #insert synonym rep_nums into sequences
    seq1 = seqToSeqSyn(seq1, syn)
    seq2 = seqToSeqSyn(seq2, syn)

    return seqToAlign(seq1, seq2, matchScore, mismatchScore, gapScore)


'''
TODO: Strategy for finding piece of text that is matched:
	the alignment object has the two alignments that match. You can just
	take out the alignment symbol from these lists and then match against the 
	original piece of text to find the beginning and ending using the Synonyms object
	to convert back to original sequence. 

'''

def check_alignment(text1, text2, align_fcn): #TODO: currently not written to use word2vec
    '''
    args: 
        text1: string
        text2: string
        align_fcn: alignment function to apply to the text
    Returns:
        prints best alignment, score, surrounding text for given alignment
    Description:
        function for examining matches produced by alignment function
    '''





if __name__ == '__main__':

    print "loading data"
    #urls to known matches
    matches = ['http://www.legislature.mi.gov/(S(ntrjry55mpj5pv55bv1wd155))/documents/2005-2006/billintroduced/House/htm/2005-HIB-5153.htm'
                'http://www.schouse.gov/sess116_2005-2006/bills/4301.htm',
                'http://www.legis.state.wv.us/Bill_Text_HTML/2008_SESSIONS/RS/Bills/hb2564%20intr.html'
                'http://www.lrc.ky.gov/record/06rs/SB38.htm',
                'http://www.okhouse.gov/Legislation/BillFiles/hb2615cs%20db.PDF',
                'https://docs.legis.wisconsin.gov/2011/related/proposals/ab69',
                'http://legisweb.state.wy.us/2008/Enroll/HB0137.pdf',
                'http://www.kansas.gov/government/legislative/bills/2006/366.pdf',
                'http://billstatus.ls.state.ms.us/documents/2006/pdf/SB/2400-2499/SB2426SG.pdf']

    #urls to nonmatches
    nonMatches = ['http://www.alec.org/model-legislation/21st-century-commercial-nexus-act/',
                    'http://www.alec.org/model-legislation/72-hour-budget-review-act/',
                    'http://www.alec.org/model-legislation/affordable-baccalaureate-degree-act/',
                    'http://www.alec.org/model-legislation/agriculture-bio-security-act/',
                    'http://www.alec.org/model-legislation/alternative-certification-act/',
                    'http://www.alec.org/model-legislation/anti-phishing-act/']

    #create id dictionary
    k = 0
    ids = {}
    while matches != []:
        ids[k] = {}
        ids[k]['url']  = matches.pop()
        ids[k]['match'] = 1 #is stand your ground act
        k += 1

    while nonMatches != []:
        ids[k] = {}
        ids[k]['url']  = nonMatches.pop()
        ids[k]['match'] = 0 #not stand your ground act
        k += 1

    keys_to_delete = [] 
    for key, value in ids.iteritems():    
        try:
            doc = urllib2.urlopen(value['url']).read()
            ids[key]['text'] = parser.from_buffer(doc)['content']
        except:
            keys_to_delete.append(key)

    #delete keys with broken urls
    for key in keys_to_delete:
        del ids[key]


    #get text from all of these docs
    #flatten sentences
    texts = [value['text'] for (key,value) in ids.iteritems()]
    text = ''
    for t in texts:
        text += t
    sentences = sent_tokenize(text)
    sentences = [s.encode('utf-8').split() for s in sentences]

    #run word2vec
    print "training word2vec"
    model = Word2Vec(sentences)


