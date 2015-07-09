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
# from nltk.tokenize import sent_tokenize

#other
import numpy as np
from collections import defaultdict

import urllib2
from tika import parser
import nltk

import json

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
        o/w returns list of tuples with score and top alignments
    Description:
        helper function for finding alignments given a list of words
    '''
    # Create a vocabulary and encode the sequences.
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
    # if return_tup == 1:
    #     return [(a.score, a.first, a.second) for a in alignments]
    # else:
    #     return Alignment(score, alignments)

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


def cleanAlignment(alignment, gap = '-'):
    '''
    arg:
        alignment object
    returns:
        2 list of alignment words without the alignment symbol

    '''
    keep1 = []
    keep2 = []
    for item in alignment[1]:
        if item != gap:
            keep1.append(item)

    for item in alignment[2]:
        if item != gap:
            keep2.append(item)

    return (keep1, keep2)

def contains(s,q):
    '''
    is the list s contained in q in order and if it is what are indices
    '''
    # largest = 0
    # start = 0
    # end = 0
    for i in range(len(q)):
        T = True
        for j in range(len(s)):
            # if largest < j:
            #     start = i
            #     end = i +j
            #     largest = end - start
            # print "largest: " + str(largest)
            # print "j: " + str(j)
            # print "i: " + str(i)
            if s[j] != q[i+j]:
                T = False
                break
        if T:
            return (i, i + j + 1)
    return (0,0)


def diff(alignment, gap = '-'):
    a = alignment[1]
    b = alignment[2]
    length = max(len(alignment[1]), len(alignment[2]))

    diff = []
    for i in range(length):
        if a[i] == b[i] or a[i] == gap or b[i] == gap:
            continue
        else:
            diff.append((a[i], b[i]))

    return diff

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
    return seqToAlign(nltk.word_tokenize(text1), nltk.word_tokenize(text2), matchScore , mismatchScore, gapScore)


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
    seq1 = nltk.word_tokenize(text1)
    seq2 = nltk.word_tokenize(text2)

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


#################
#Evaluation Functions

def create_bills(ls):
    '''
    args:
        ls: list of lists of urls that correspond to matches

    returns:
        dictionary grouped by matches
    '''
    k = 0
    bill_id = 0
    bills = {}
    for urls in ls:
        for url in urls:
            try:
                print "bill_id: " + str(bill_id)
                bills[bill_id] = {}
                doc = urllib2.urlopen(url).read()
                text = parser.from_buffer(doc)['content']

                print text
                
                bills[bill_id]['url'] = url
                bills[bill_id]['text'] = text
                bills[bill_id]['match'] = k
            except:
                pass
            bill_id += 1
        k += 1

    try:
        for bill in bills.keys():
            if bills[bill]['text'] == '':
                del bills[bill]
    except:
        pass

    return bills


def check_alignment(text1, text2, align_fcn, surround_text = 6): #TODO: currently not written to use word2vec or to input parameters
    '''
    args: 
        text1: string
        text2: string
        align_fcn: alignment function to apply to the text
        surround_text: number of words to include before and after alignment in original text
    Returns:
        prints best alignment, score, surrounding text for given alignment
    Description:
        function for examining matches of pair of texts produced by alignment function
    '''
    aligns = align_fcn(text1, text2)
    alignment = aligns[0]
    seq1 = nltk.word_tokenize(text1)
    seq2 = nltk.word_tokenize(text2)

    align_clean_1, align_clean_2 = cleanAlignment(alignment)

    [i,j] = contains(align_clean_1, seq1)
    [k,l] = contains(align_clean_2, seq2)

    print "i,j: " + str((i,j))
    print "k,l: " + str((k,l))

    print "alignment score: " + str(alignment[0])
    print '\n'
    
    print "first alignment: " + '\n' + ' '.join(align_clean_1)
    print '\n'

    print "first alignment text: " + '\n' + ' '.join(seq1[max(i-surround_text, 0): j+surround_text])
    print '\n'

    print "second alignment: " + '\n' + ' '.join(align_clean_2)
    print '\n'    

    print "second alignment text: " + '\n' + ' '.join(seq2[max(k-surround_text, 0): l+surround_text])
    print '\n'

    print diff(alignment)

    # return alignment


def evaluate_algorithm(evals):
    '''
    args:
        matches: dictionary with field corresponding to text and match groups

    returns:
        matrix with scores between all pairs and a dictionary with information
    '''
    scores = np.zeros((max(evals.keys())+1, max(evals.keys())+1))

    results = {}

    for i in evals.keys():
        for j in evals.keys():
            if i < j: #local alignment gives symmetric distance
                text1 = evals[i]['text']
                text2 = evals[j]['text']

                # Create sequences to be aligned.
                alignments = align(evals)

                scores[i,j] = alignment[0][0]

                results[(i,j)] ={}
                results[(i,j)]['alignments'] = [(alignment[1], alignment[2]) for alignment in alignments]

                results[(i,j)] = diff(alignment[1], alignment[2])

    return scores, results


def plot_scores(scores, evals):

    matchScores = []
    nonMatchScores = []

    for i in evals.keys():
        for j in evals.keys():
            if scores[i,j] == 0:
                #ignore if score zero because url is broken
                pass
            elif i < j and evals[i]['match'] == evals[j]['match']:
                matchScores.append(scores[i,j])
            else:
                nonMatchScores.append(scores[i,j])

    val = 0. 
    plt.plot(matchScores, np.zeros_like(matchScores), 'o')
    plt.plot(nonMatchScores, np.zeros_like(nonMatchScores), 'x')
    plt.plot()
    plt.show()

def main():
    print "loading data"

    bills = json.loads('/Users/jkatzsamuels/Desktop/dssg/sunlight/policy_diffusion/data/eval.json')

    scores, results = evaluate_algorithm(bills)

    plot_scores(scores, evals) 

    #urls to known matches
    # matches = ['http://www.legislature.mi.gov/(S(ntrjry55mpj5pv55bv1wd155))/documents/2005-2006/billintroduced/House/htm/2005-HIB-5153.htm'
    #             'http://www.schouse.gov/sess116_2005-2006/bills/4301.htm',
    #             'http://www.legis.state.wv.us/Bill_Text_HTML/2008_SESSIONS/RS/Bills/hb2564%20intr.html'
    #             'http://www.lrc.ky.gov/record/06rs/SB38.htm',
    #             'http://www.okhouse.gov/Legislation/BillFiles/hb2615cs%20db.PDF',
    #             'https://docs.legis.wisconsin.gov/2011/related/proposals/ab69',
    #             'http://legisweb.state.wy.us/2008/Enroll/HB0137.pdf',
    #             'http://www.kansas.gov/government/legislative/bills/2006/366.pdf',
    #             'http://billstatus.ls.state.ms.us/documents/2006/pdf/SB/2400-2499/SB2426SG.pdf']

    #urls to nonmatches
    # nonMatches = ['http://www.alec.org/model-legislation/21st-century-commercial-nexus-act/',
    #                 'http://www.alec.org/model-legislation/72-hour-budget-review-act/',
    #                 'http://www.alec.org/model-legislation/affordable-baccalaureate-degree-act/',
    #                 'http://www.alec.org/model-legislation/agriculture-bio-security-act/',
    #                 'http://www.alec.org/model-legislation/alternative-certification-act/',
    #                 'http://www.alec.org/model-legislation/anti-phishing-act/']

    #create id dictionary
    # k = 0
    # ids = {}
    # while matches != []:
    #     ids[k] = {}
    #     ids[k]['url']  = matches.pop()
    #     ids[k]['match'] = 1 #is stand your ground act
    #     k += 1

    # while nonMatches != []:
    #     ids[k] = {}
    #     ids[k]['url']  = nonMatches.pop()
    #     ids[k]['match'] = 0 #not stand your ground act
    #     k += 1

    # keys_to_delete = [] 
    # for key, value in ids.iteritems():    
    #     try:
    #         doc = urllib2.urlopen(value['url']).read()
    #         ids[key]['text'] = parser.from_buffer(doc)['content']
    #     except:
    #         keys_to_delete.append(key)

    # #delete keys with broken urls
    # for key in keys_to_delete:
    #     del ids[key]

    # text1 = ids[2]['text']
    # text2 = ids[3]['text']

    # for i in ids.keys():
    #     for j in ids.keys():
    #         if i < j:
    #             text1 = ids[i]['text'] 
    #             text2 = ids[j]['text']

    #             check_alignment(text1,text2)
    #             raw_input("Press Enter to continue...")
   # similar_bills = [['http://www.azleg.gov/legtext/52leg/1r/bills/hb2505p.pdf',
   #      'http://www.legis.state.ak.us/basis/get_bill_text.asp?hsid=SB0012B&session=29',
   #      'http://www.capitol.hawaii.gov/session2015/bills/HB9_.PDF',
   #      'http://www.capitol.hawaii.gov/session2015/bills/HB1047_.PDF',
   #      'http://flsenate.gov/Session/Bill/2015/1490/BillText/Filed/HTML',
   #      'http://ilga.gov/legislation/fulltext.asp?DocName=09900SB1836&GA=99&SessionId=88&DocTypeId=SB&LegID=88673&DocNum=1836&GAID=13&Session=&print=true'
   #      'http://www.legis.la.gov/Legis/ViewDocument.aspx?d=933306',
   #      'http://mgaleg.maryland.gov/2015RS/bills/sb/sb0040f.pdf',
   #      'http://www.legislature.mi.gov/documents/2015-2016/billintroduced/House/htm/2015-HIB-4167.htm',
   #      'https://www.revisor.mn.gov/bills/text.php?number=HF549&version=0&session=ls89&session_year=2015&session_number=0',
   #      'http://www.njleg.state.nj.us/2014/Bills/A2500/2354_R2.HTM',
   #      'http://assembly.state.ny.us/leg/?sh=printbill&bn=A735&term=2015',
   #      'http://www.ncga.state.nc.us/Sessions/2015/Bills/House/HTML/H270v1.html',
   #      'https://olis.leg.state.or.us/liz/2015R1/Downloads/MeasureDocument/HB2005/A-Engrossed',
   #      'https://olis.leg.state.or.us/liz/2015R1/Downloads/MeasureDocument/SB947/Introduced',
   #      'http://www.legis.state.pa.us/CFDOCS/Legis/PN/Public/btCheck.cfm?txtType=HTM&sessYr=2015&sessInd=0&billBody=H&billTyp=B&billNbr=0624&pn=0724',
   #      'http://www.scstatehouse.gov/sess121_2015-2016/prever/172_20141203.htm',
   #      'http://lawfilesext.leg.wa.gov/Biennium/2015-16/Htm/Bills/House%20Bills/1356.htm',
   #      'http://www.legis.state.wv.us/Bill_Status/bills_text.cfm?billdoc=hb2874%20intr.htm&yr=2015&sesstype=RS&i=2874',
   #      'http://www.legis.state.wv.us/Bill_Status/bills_text.cfm?billdoc=hb2874%20intr.htm&yr=2015&sesstype=RS&i=2874'],
   #      ['ftp://ftp.cga.ct.gov/2015/tob/h/2015HB-06784-R00-HB.htm',
   #      'http://www.capitol.hawaii.gov/session2015/bills/SB129_.PDF'],
   #      ['http://nebraskalegislature.gov/FloorDocs/104/PDF/Intro/LB493.pdf',
   #      'http://www.gencourt.state.nh.us/legislation/2015/HB0600.html'],
   #      ['http://alecexposed.org/w/images/2/2d/7K5-No_Sanctuary_Cities_for_Illegal_Immigrants_Act_Exposed.pdf',
   #      'http://www.kslegislature.org/li_2012/b2011_12/measures/documents/hb2578_00_0000.pdf',
   #      'http://flsenate.gov/Session/Bill/2011/0237/BillText/Filed/HTML',
   #      'http://openstates.org/al/bills/2012rs/SB211/',
   #      'http://le.utah.gov/~2011/bills/static/HB0497.html',
   #      'http://webserver1.lsb.state.ok.us/cf_pdf/2013-14%20FLR/HFLR/HB1436%20HFLR.PDF'],
   #      ['http://www.alec.org/model-legislation/the-disclosure-of-hydraulic-fracturing-fluid-composition-act/',
   #      'ftp://ftp.legis.state.tx.us/bills/82R/billtext/html/house_bills/HB03300_HB03399/HB03328S.htm'],
   #      ['http://www.legislature.mi.gov/(S(ntrjry55mpj5pv55bv1wd155))/documents/2005-2006/billintroduced/House/htm/2005-HIB-5153.htm',
   #      'http://www.schouse.gov/sess116_2005-2006/bills/4301.htm',
   #      'http://www.lrc.ky.gov/record/06rs/SB38.htm',
   #      'http://www.okhouse.gov/Legislation/BillFiles/hb2615cs%20db.PDF',
   #      'http://state.tn.us/sos/acts/105/pub/pc0210.pdf',
   #      'https://docs.legis.wisconsin.gov/2011/related/proposals/ab69',
   #      'http://legisweb.state.wy.us/2008/Enroll/HB0137.pdf',
   #      'http://www.kansas.gov/government/legislative/bills/2006/366.pdf',
   #      'http://billstatus.ls.state.ms.us/documents/2006/pdf/SB/2400-2499/SB2426SG.pdf'],
   #      ['http://www.alec.org/model-legislation/state-withdrawal-from-regional-climate-initiatives/',
   #      'http://www.legislature.mi.gov/documents/2011-2012/resolutionintroduced/House/htm/2011-HIR-0134.htm',
   #      'http://www.nmlegis.gov/Sessions/11%20Regular/memorials/house/HJM024.html'],
   #      ['http://alecexposed.org/w/images/9/90/7J1-Campus_Personal_Protection_Act_Exposed.pdf',
   #      'ftp://ftp.legis.state.tx.us/bills/831/billtext/html/house_bills/HB00001_HB00099/HB00056I.htm'],
   #      ['http://essexuu.org/ctstat.html',
   #      'http://alisondb.legislature.state.al.us/alison/codeofalabama/constitution/1901/CA-170364.htm'],
   #      ['http://www.legis.state.ak.us/basis/get_bill_text.asp?hsid=HB0162A&session=27'
   #      'https://legiscan.com/AL/text/HB19/id/327641/Alabama-2011-HB19-Enrolled.pdf',
   #      'http://www.leg.state.co.us/clics/clics2012a/csl.nsf/fsbillcont3/0039C9417C9D9D5D87257981007F3CC9?open&file=1111_01.pdf',
   #      'http://www.capitol.hawaii.gov/session2012/Bills/HB2221_.PDF',
   #      'http://ilga.gov/legislation/fulltext.asp?DocName=09700HB3058&GA=97&SessionId=84&DocTypeId=HB&LegID=60409&DocNum=3058&GAID=11&Session=&print=true',
   #      'http://coolice.legis.iowa.gov/Legislation/84thGA/Bills/SenateFiles/Introduced/SF142.html',
   #      'ftp://www.arkleg.state.ar.us/Bills/2011/Public/HB1797.pdf',
   #      'http://billstatus.ls.state.ms.us/documents/2012/html/HB/0900-0999/HB0921SG.htm',
   #      'http://www.leg.state.nv.us/Session/76th2011/Bills/SB/SB373.pdf',
   #      'http://www.njleg.state.nj.us/2012/Bills/A1000/674_I1.HTM',
   #      'http://webserver1.lsb.state.ok.us/cf_pdf/2011-12%20INT/hB/HB2821%20INT.PDF',
   #      'http://www.legis.state.pa.us/CFDOCS/Legis/PN/Public/btCheck.cfm?txtType=PDF&sessYr=2011&sessInd=0&billBody=H&billTyp=B&billNbr=0934&pn=1003',
   #      'http://www.capitol.tn.gov/Bills/107/Bill/SB0016.pdf'],
   #      ['http://www.legislature.idaho.gov/idstat/Title39/T39CH6SECT39-608.htm',
   #      'http://www.legis.nd.gov/cencode/t12-1c20.pdf?20150708171557']
   #      ]




if __name__ == '__main__':
    main()

               


#     #get text from all of these docs
#     #flatten sentences
#     texts = [value['text'] for (key,value) in ids.iteritems()]
#     text = ''
#     for t in texts:
#         text += t
#     sentences = sent_tokenize(text)
#     sentences = [s.encode('utf-8').split() for s in sentences]

#     #run word2vec
#     print "training word2vec"
#     model = Word2Vec(sentences)






