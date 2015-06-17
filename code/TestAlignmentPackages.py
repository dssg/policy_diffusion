##Test Biopython

import urllib2
from tika import parser
	
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

import networkx as nx

'''
Test packages for local alignment with stand your ground laws.

'''


##########Note: I need to make a nested dictionary. look it up on stack overflow

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

#urls to matches
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
        ids[key]['doc_type'] = parser.from_buffer(doc)['Content-Type']
    except:
        keys_to_delete.append(key)

#delete keys with broken urls
for key in keys_to_delete:
    del ids[key]

###############################
###alignment
from alignment.sequence import Sequence
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner, LocalSequenceAligner

scores = np.zeros((max(ids.keys())+1, max(ids.keys())+1))

alignments = {} #stores text that matches by comparison key = (i,j)

for i in ids.keys():
    for j in ids.keys():
        if i < j: #local alignment gives symmetric distance
            text1 = ids[i]['text']
            text2 = ids[j]['text']

            # Create sequences to be aligned.
            a = Sequence(text1.split())
            b = Sequence(text2.split())

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
                score = alignment.score
                if scores[i,j] <= score:
                    print str((i,j)) + "-score: " + str(score)
                    scores[i,j] = score

                    # print alignment
                    alignments[(i,j)] = alignment 

                # print alignment
                # print 'Alignment score:', alignment.score
                # print 'Percent identity:', alignment.percentIdentity()


#plot scores
matchScores = []
nonMatchScores = []
for i in ids.keys():
    for j in ids.keys():
        if scores[i,j] == 0:
            #ignore if score zero because url is broken
            pass
        elif i < j and ids[i]['match'] == 1 and ids[j]['match'] == 1:
            matchScores.append(scores[i,j])
        elif ids[i]['match'] == 0 and ids[j]['match'] == 0:
            #ignore if both from alec because we don't care
            pass
        else:
            nonMatchScores.append(scores[i,j])

val = 0. 
plt.plot(matchScores, np.zeros_like(matchScores), 'o')
plt.plot(nonMatchScores, np.zeros_like(nonMatchScores), 'x')
plt.plot()
plt.show()

###############################
#create network graph visualization

G=nx.Graph()
for key in ids.keys():
    G.add_node(key, match = ids[key]['match'])

for i in ids.keys():
    for j in ids.keys():
        if scores[i,j] == 0:
            #ignore if score zero because url is broken
            pass
        elif ids[i]['match'] == 0 and ids[j]['match'] == 0:
            #ignore if both from alec because we don't care
            pass
        else:
            G.add_edge(i,j, weight = scores[i,j])

#only look at matches
for i, d in G.nodes(data = True):
    if d['match'] != 1:
        G.remove_node(i)

edges = [(u,v,d['weight']) for (u,v,d) in G.edges(data=True)]

# sorted(edges, key=lambda x: x[2])

pos = nx.spring_layout(G)

# nodes
nx.draw_networkx_nodes(G,pos,node_size=700)

# edges
for (u,v,weight) in edges:
    nx.draw_networkx_edges(G,pos,edgelist=[(u,v)], alpha = min(weight / 100.,1))


# labels
nx.draw_networkx_labels(G,pos, font_size=20,font_family='sans-serif')

plt.show()


# #################################
# ###bio
# ##beware: this is slow/makes my terminal crash
# from Bio.Alphabet import SingleLetterAlphabet
# from Bio.Seq import Seq

# matrix = matlist.blosum62
# gap_open = -10
# gap_extend = -0.5

# test1 = "hellothere"
# test2 = "whatishello"

# alns = pairwise2.align.globalxx(test1, test2)

# alns = pairwise2.align.globalxx(text1, text2)

# top_aln = alns[0]
