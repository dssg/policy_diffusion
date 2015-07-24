from __future__ import division
import numpy as np
from numba import jit
import itertools
import time
import matplotlib.pyplot as plt
import sys
import abc
import json
import pickle
from tika import parser
import urllib2
import re
import pandas as pd
from sklearn.decomposition import PCA
from fast_alignment import *    


############################################################
##evaluation functions

def evaluate_algorithm(bills, match_score = 3, mismatch_score = -1, gap_score = -2):
    '''
    args:
        matches: dictionary with field corresponding to text and match groups

    returns:
        matrix with scores between all pairs and a dictionary with information
    '''
    scores = np.zeros((max(bills.keys())+1, max(bills.keys())+1))

    results = {}

    for i in bills.keys():
        for j in bills.keys():
            if i < j: #local alignment gives symmetric distance
                if bills[i] == {} or bills[j] == {}:
                    continue

                text1 = bills[i]['text'].split()
                text2 = bills[j]['text'].split()

                if text1 == '' or text2 == '':
                    continue

                # Create sequences to be aligned.
                f = LocalAlignment(text1, text2)
                alignments = f.align(match_score, mismatch_score, gap_score)

                scores[i,j] = alignments[0][0]

                results[(i,j)] ={}
                results[(i,j)]['alignments'] = [(alignment[1], alignment[2]) for alignment in alignments]
                results[(i,j)]['score'] = alignments[0][0]
                results[(i,j)]['match'] = (bills[i]['match'] == bills[j]['match'])

                results[(i,j)]['diff'] = diff(alignment)

                print 'i: ' + str(i) + ', j: ' + str(j) + ' score: ' + str(alignments[0][0])

    return scores, results


def plot_scores(scores, bills):

    matchScores = []
    nonMatchScores = []

    for i in bills.keys():
        for j in bills.keys():

            if scores[i,j] == 0:
                #ignore if score zero because url is broken
                pass
            elif i < j and bills[i]['match'] == bills[j]['match']:
                matchScores.append(min(scores[i,j],200))
            else:
                nonMatchScores.append(min(scores[i,j],200))

    bins = np.linspace(min(nonMatchScores + matchScores), max(nonMatchScores + matchScores), 100)
    plt.hist(nonMatchScores, bins, alpha=0.5, label='Non-Matches')
    plt.hist(matchScores, bins, alpha=0.5, label='Matches')
    plt.legend(loc='upper right')
    plt.xlabel('Alignment Score')
    plt.ylabel('Number of Bill Pairs')
    plt.title('Distribution of Alignment Scores')
    plt.show()

    #make boxplot
    data_to_plot = [matchScores, nonMatchScores]
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(data_to_plot)
    ax.set_xticklabels(['Match Scores', 'Non-Match Scores'])
    fig.show()


def grid_search(bills, match_scores = [2,3,4,5], mismatch_scores = [-1,-2,-3,-4,-5], gap_scores = [-1,-2,-3,-4,-5]):
    grid_performance = {}
    for match_score in match_scores:
        for mismatch_score in mismatch_scores:
            for gap_score in gap_scores:

                print 'running model: match_score {0} mismatch_score {1} gap_score {2}'.format(match_score, mismatch_score, gap_score)

                grid_performance[(match_score, mismatch_score, gap_score)] = {}
                scores, results = evaluate_algorithm(bills, match_score, mismatch_score, gap_score)
                grid_performance[(match_score, mismatch_score, gap_score)]['results'] = results
                grid_performance[(match_score, mismatch_score, gap_score)]['scores'] = scores

    return grid_performance

def grid_to_df(grid):
    t = []
    for key1, value1 in grid.items():
        for key2, value2 in value1['results'].items():
            t.append(list(key1) + [key2, value2['score'], value2['match']])
    
    df = pd.DataFrame(t)
    df.columns = ['match_score', 'mismatch_score', 'gap_score', 'pair', 'score', 'match']

    return df

def plot_grid(df):

    #make maximum possible 500
    df.loc[df['score']>500,'score'] = 500

    #match plot
    df_match = df[(df['mismatch_score'] == -2) & (df['gap_score'] == -1)]

    g = sns.FacetGrid(df_match, col="match_score")
    g = g.map(sns.boxplot, "match", "score")
    sns.plt.ylim(0,400)
    sns.plt.show()

    #mismatch plot
    df_mismatch = df[(df['match_score'] == 3) & (df['gap_score'] == -1)]

    g = sns.FacetGrid(df_mismatch, col="mismatch_score")
    g = g.map(sns.boxplot, "match", "score")
    sns.plt.ylim(0,400)
    sns.plt.show()

    #gap plot
    df_gap = df[(df['match_score'] == 3) & (df['mismatch_score'] == -2)]

    g = sns.FacetGrid(df_gap, col="gap_score")
    g = g.map(sns.boxplot, "match", "score")
    sns.plt.ylim(0,400)
    sns.plt.show()

def evaluate():
    bills = load_bills()

    scores, results = evaluate_algorithm(bills)

    plot_scores(scores, bills) 

    return scores, results


def inspect_alignments(results, match_type = 0, start_score = 'max'):
        '''
            match_type is 0 if you want to inspect non-matches
            and 1 if you want to inspect matches
        '''
        alignments = [tuple([value['score']] + list(value['alignments'][0]) + [key])  for key, value in results.items() if value['match'] == match_type]
        sorted_alignments = sorted(alignments, key=lambda tup: tup[0], reverse = True)

        if start_score == 'max':
            for score, a1, a2, key in sorted_alignments:
                for i in range(len(a1)):
                    #note a1 and a2 must have same length
                    print a1[i], a2[i]
                print 'score: ', score
                print 'key: ', key
                raw_input("Press Enter to continue...")
        else:
            for score, a1, a2, key in sorted_alignments:
                if score > start_score:
                    pass
                else:
                    for i in range(len(a1)):
                        print a1[i], a2[i]
                    print 'score: ', score
                    print 'key: ', key
                    raw_input("Press Enter to continue...")

############################################################
##alignments utils
def diff(alignment):
    a = alignment[1]
    b = alignment[2]
    length = min(len(alignment[1]), len(alignment[2]))

    diff = []
    for i in range(length):
        if a[i] == b[i] or a[i] == '-' or b[i] == '-':
            continue
        else:
            diff.append((a[i], b[i]))

    return diff


############################################################
##Alignment Feature Generation and Plotting

def alignment_features(left, right):
    '''
    This function takes as input two alignments and produce features of these
    '''
    #alignment features
    features = {}
    features['length'] = len(left)
    features['num_gaps_l'] = 0
    features['num_gaps_r'] = 0
    features['num_mismatches'] = 0
    features['num_matches'] = 0
    features['avg_gap_length_l'] = []
    features['avg_gap_length_r'] = []
    features['avg_consec_match_length'] = []

    #helper variables
    prev_gap_l = False
    prev_gap_r = False
    prev_match = False
    for i in range(len(left)):
        # print 'i: ', i
        # print 'features: ', features
        if left[i] == '-':
            features['num_gaps_l'] += 1
            if not prev_gap_l:
                features['avg_gap_length_l'].append(1)
                prev_gap_l = True
            else:
                features['avg_gap_length_l'][-1] += 1
        else:
            prev_gap_l = False
        if right[i] == '-':
            features['num_gaps_r'] += 1
            if not prev_gap_r:
                features['avg_gap_length_r'].append(1)
                prev_gap_r = True
            else:
                features['avg_gap_length_r'][-1] += 1
        else:
            prev_gap_r = False
        if left[i] != '-' and right[i] != '-':
            if left[i] != right[i]:
                features['num_mismatches'] += 1
            elif left[i] == right[i]:
                features['num_matches'] += 1
                if not prev_match:
                    features['avg_consec_match_length'].append(1)
                    prev_match = True
                else:
                    features['avg_consec_match_length'][-1] += 1
        if left[i] != right[i]:
            prev_match = False

    if features['avg_gap_length_l'] != []:
        features['avg_gap_length_l'] = np.mean(features['avg_gap_length_l'])
    else:
        features['avg_gap_length_l'] = 0
    if features['avg_gap_length_r'] != []:
        features['avg_gap_length_r'] = np.mean(features['avg_gap_length_r'])
    else:
        features['avg_gap_length_r'] = 0
    if features['avg_consec_match_length'] != []:
        features['avg_consec_match_length'] = np.mean(features['avg_consec_match_length'])
    else:
        features['avg_consec_match_length'] = 0

    return features

#good test case: results[(21,22)]
def test_alignment_features():

    def true_features(length, num_gaps_l, num_gaps_r, num_mismatches, 
                        num_matches, avg_gap_length_l, avg_gap_length_r,
                        avg_consec_match_length):
        truth = {}
        truth['length'] = length
        truth['num_gaps_l'] = num_gaps_l
        truth['num_gaps_r'] = num_gaps_r
        truth['num_mismatches'] = num_mismatches
        truth['num_matches'] = num_matches
        truth['avg_gap_length_l'] = avg_gap_length_l
        truth['avg_gap_length_r'] = avg_gap_length_r
        truth['avg_consec_match_length'] = avg_consec_match_length

        return truth

    def check_features_truth(truth, features):
        for field in features.keys():
            if features[field] != truth[field]:
                print field + ' is inconsistent'


    a1 = range(1,10) + ['-', '-'] + range(12,15)
    a2 = range(1,4) + ['-', '-', 7] + range(7,15)

    features = alignment_features(a1, a2)
    truth = true_features(14,2,2,1,9,2,2,3)

    print 'first test: '
    check_features_truth(truth, features)

    b1 = range(1,10)
    b2 = ['-','-',3,5,6,7, '-', '-', 9]
    features = alignment_features(b1, b2)
    truth = true_features(9,0,4,3,2,0,2,1)

    print 'second test: '
    check_features_truth(truth, features)


def calc_pop_results(results):
    for key,value in results.iteritems():
        left, right = value['alignments'][0]
        features = alignment_features(left, right)
        results[key]['features'] = features

    return results

def low_rank_plot(results):

    #convert dictionary to matrix
    matches = [[value for key, value in values['features'].items()] \
        for keys, values in results.items() if values['match'] == 1]

    non_matches = [[value for key, value in values['features'].items()] \
        for keys, values in results.items() if values['match'] == 0]

    #matches from 0 to match_index
    match_index = len(matches)

    data = np.array(matches + non_matches)

    sklearn_pca = PCA(n_components=2)
    sklearn_transf = sklearn_pca.fit_transform(data)

    #plot data
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.plot(sklearn_transf[:match_index,0],sklearn_transf[:match_index,1], 
             'o', markersize=7, color='blue', alpha=0.5, label='matches')
    plt.plot(sklearn_transf[match_index:,0], sklearn_transf[match_index:,1], 
             '^', markersize=7, color='red', alpha=0.5, label='non-matches')
    plt.xlim([-50,1000])
    plt.ylim([-500,500])
    plt.legend(loc='upper left');
    plt.show()


def plot_alignment_stats(results):
    matches = [value['features'] for key, value in results.items() if value['match'] == 1]
    non_matches = [value['features'] for key, value in results.items() if value['match'] == 0]
    
    fields = matches[0].keys()
    for field in fields:
        matches_field = [item[field] for item in matches]
        non_matches_field = [item[field] for item in non_matches]

        bins = np.linspace(min(matches_field + non_matches_field), max(matches_field + non_matches_field), 100)
        plt.hist(matches_field, bins, alpha=0.5, label='Matches')
        plt.hist(non_matches_field, bins, alpha=0.5, label='Non-Matches')
        plt.legend(loc='upper right')
        plt.title(field)
        plt.show()

        data_to_plot = [matches_field, non_matches_field]
        fig = plt.figure(1, figsize=(9, 6))
        ax = fig.add_subplot(111)
        bp = ax.boxplot(data_to_plot)
        plt.ylim([0,50])
        ax.set_xticklabels(['Matches', 'Non-Matches'])
        plt.title(field)
        plt.show()


#data creating, saving, and loading
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
    bad_count = 0
    for urls in ls:
        for url in urls:
            try:
                print "bill_id: " + str(bill_id)
                bills[bill_id] = {}
                doc = urllib2.urlopen(url).read()
                text = parser.from_buffer(doc)['content']

                #clean up text
                cleaned_text = clean_text(text)
                cleaned_text_list = cleaned_text.split('\n')

                #delete lines with just number
                re_string = '\\n\s[0-9][0-9]|\\n[0-9][0-9]|\\n[0-9]|\\n\s[0-9]'
                cleaned_text_list = [re.sub(re_string,'',t) for t in cleaned_text_list]

                #delete empty lines
                cleaned_text_list = [re.sub( '\s+', ' ', x) for x in cleaned_text_list]
                cleaned_text_list = [x for x in cleaned_text_list if x is not None and len(x)>2]

                cleaned_text = ' '.join(cleaned_text_list)

                print cleaned_text
                
                bills[bill_id]['url'] = url
                bills[bill_id]['text'] = cleaned_text
                bills[bill_id]['match'] = k
            except:
                pass
                bad_count += 1
                print 'bad_count: ', bad_count
            bill_id += 1
        k += 1

    try:
        for bill in bills.keys():
            if bills[bill]['text'] == '':
                del bills[bill]
    except:
        pass

    return bills


def create_save_bills(bill_list):
    bills = create_bills(bill_list)
    with open('../data/bills.json', 'wb') as fp:
        pickle.dump(bills, fp)


def load_bills():
    with open('../data/bills.json','rb') as fp:
        bills =pickle.load(fp)

    return bills


def load_scores():
    scores = np.load('../data/scores.npy')

    return scores


def save_results(results):
    with open('../data/results.json','wb') as fp:
        pickle.dump(results, fp)


def load_results():
    with open('../data/results.json','rb') as fp:
        data =pickle.load(fp)
    return data


if __name__ == '__main__':
    #each list in this list of lists contains bills that are matches
    similar_bills = [['http://www.azleg.gov/legtext/52leg/1r/bills/hb2505p.pdf',
    'http://www.legis.state.ak.us/basis/get_bill_text.asp?hsid=SB0012B&session=29',
    'http://www.capitol.hawaii.gov/session2015/bills/HB9_.PDF',
    'http://www.capitol.hawaii.gov/session2015/bills/HB1047_.PDF',
    'http://flsenate.gov/Session/Bill/2015/1490/BillText/Filed/HTML',
    'http://ilga.gov/legislation/fulltext.asp?DocName=09900SB1836&GA=99&SessionId=88&DocTypeId=SB&LegID=88673&DocNum=1836&GAID=13&Session=&print=true'
    'http://www.legis.la.gov/Legis/ViewDocument.aspx?d=933306',
    'http://mgaleg.maryland.gov/2015RS/bills/sb/sb0040f.pdf',
    'http://www.legislature.mi.gov/documents/2015-2016/billintroduced/House/htm/2015-HIB-4167.htm',
    'https://www.revisor.mn.gov/bills/text.php?number=HF549&version=0&session=ls89&session_year=2015&session_number=0',
    'http://www.njleg.state.nj.us/2014/Bills/A2500/2354_R2.HTM',
    'http://assembly.state.ny.us/leg/?sh=printbill&bn=A735&term=2015',
    'http://www.ncga.state.nc.us/Sessions/2015/Bills/House/HTML/H270v1.html',
    'https://olis.leg.state.or.us/liz/2015R1/Downloads/MeasureDocument/HB2005/A-Engrossed',
    'https://olis.leg.state.or.us/liz/2015R1/Downloads/MeasureDocument/SB947/Introduced',
    'http://www.legis.state.pa.us/CFDOCS/Legis/PN/Public/btCheck.cfm?txtType=HTM&sessYr=2015&sessInd=0&billBody=H&billTyp=B&billNbr=0624&pn=0724',
    'http://www.scstatehouse.gov/sess121_2015-2016/prever/172_20141203.htm',
    'http://lawfilesext.leg.wa.gov/Biennium/2015-16/Htm/Bills/House%20Bills/1356.htm',
    'http://www.legis.state.wv.us/Bill_Status/bills_text.cfm?billdoc=hb2874%20intr.htm&yr=2015&sesstype=RS&i=2874',
    'http://www.legis.state.wv.us/Bill_Status/bills_text.cfm?billdoc=hb2874%20intr.htm&yr=2015&sesstype=RS&i=2874',
    'ftp://ftp.cga.ct.gov/2015/tob/h/2015HB-06784-R00-HB.htm',
    'http://www.capitol.hawaii.gov/session2015/bills/SB129_.PDF',
    'http://nebraskalegislature.gov/FloorDocs/104/PDF/Intro/LB493.pdf',
    'http://www.gencourt.state.nh.us/legislation/2015/HB0600.html'],
    ['http://alecexposed.org/w/images/2/2d/7K5-No_Sanctuary_Cities_for_Illegal_Immigrants_Act_Exposed.pdf',
    'http://www.kslegislature.org/li_2012/b2011_12/measures/documents/hb2578_00_0000.pdf',
    'http://flsenate.gov/Session/Bill/2011/0237/BillText/Filed/HTML',
    'http://openstates.org/al/bills/2012rs/SB211/',
    'http://le.utah.gov/~2011/bills/static/HB0497.html',
    'http://webserver1.lsb.state.ok.us/cf_pdf/2013-14%20FLR/HFLR/HB1436%20HFLR.PDF'],
    ['http://www.alec.org/model-legislation/the-disclosure-of-hydraulic-fracturing-fluid-composition-act/',
    'ftp://ftp.legis.state.tx.us/bills/82R/billtext/html/house_bills/HB03300_HB03399/HB03328S.htm'],
    ['http://www.legislature.mi.gov/(S(ntrjry55mpj5pv55bv1wd155))/documents/2005-2006/billintroduced/House/htm/2005-HIB-5153.htm',
    'http://www.schouse.gov/sess116_2005-2006/bills/4301.htm',
    'http://www.lrc.ky.gov/record/06rs/SB38.htm',
    'http://www.okhouse.gov/Legislation/BillFiles/hb2615cs%20db.PDF',
    'http://state.tn.us/sos/acts/105/pub/pc0210.pdf',
    'https://docs.legis.wisconsin.gov/2011/related/proposals/ab69',
    'http://legisweb.state.wy.us/2008/Enroll/HB0137.pdf',
    'http://www.kansas.gov/government/legislative/bills/2006/366.pdf',
    'http://billstatus.ls.state.ms.us/documents/2006/pdf/SB/2400-2499/SB2426SG.pdf'],
    ['http://www.alec.org/model-legislation/state-withdrawal-from-regional-climate-initiatives/',
    'http://www.legislature.mi.gov/documents/2011-2012/resolutionintroduced/House/htm/2011-HIR-0134.htm',
    'http://www.nmlegis.gov/Sessions/11%20Regular/memorials/house/HJM024.html'],
    ['http://alecexposed.org/w/images/9/90/7J1-Campus_Personal_Protection_Act_Exposed.pdf',
    'ftp://ftp.legis.state.tx.us/bills/831/billtext/html/house_bills/HB00001_HB00099/HB00056I.htm'],
    ['http://essexuu.org/ctstat.html',
    'http://alisondb.legislature.state.al.us/alison/codeofalabama/constitution/1901/CA-170364.htm'],
    ['http://www.legis.state.ak.us/basis/get_bill_text.asp?hsid=HB0162A&session=27'
    'https://legiscan.com/AL/text/HB19/id/327641/Alabama-2011-HB19-Enrolled.pdf',
    'http://www.leg.state.co.us/clics/clics2012a/csl.nsf/fsbillcont3/0039C9417C9D9D5D87257981007F3CC9?open&file=1111_01.pdf',
    'http://www.capitol.hawaii.gov/session2012/Bills/HB2221_.PDF',
    'http://ilga.gov/legislation/fulltext.asp?DocName=09700HB3058&GA=97&SessionId=84&DocTypeId=HB&LegID=60409&DocNum=3058&GAID=11&Session=&print=true',
    'http://coolice.legis.iowa.gov/Legislation/84thGA/Bills/SenateFiles/Introduced/SF142.html',
    'ftp://www.arkleg.state.ar.us/Bills/2011/Public/HB1797.pdf',
    'http://billstatus.ls.state.ms.us/documents/2012/html/HB/0900-0999/HB0921SG.htm',
    'http://www.leg.state.nv.us/Session/76th2011/Bills/SB/SB373.pdf',
    'http://www.njleg.state.nj.us/2012/Bills/A1000/674_I1.HTM',
    'http://webserver1.lsb.state.ok.us/cf_pdf/2011-12%20INT/hB/HB2821%20INT.PDF',
    'http://www.legis.state.pa.us/CFDOCS/Legis/PN/Public/btCheck.cfm?txtType=PDF&sessYr=2011&sessInd=0&billBody=H&billTyp=B&billNbr=0934&pn=1003',
    'http://www.capitol.tn.gov/Bills/107/Bill/SB0016.pdf'],
    ['http://www.legislature.idaho.gov/idstat/Title39/T39CH6SECT39-608.htm',
    'http://www.legis.nd.gov/cencode/t12-1c20.pdf?20150708171557']
    ]

    with open('clean_eval_bills.json') as data_file:    
        clean_bills = json.load(data_file)

    clean_bills = {int(key) : value for key, value in clean_bills.items()}

    grid = grid_search(clean_bills)

    with open('grid_results.p', 'wb') as fp:
        pickle.dump(grid, fp)

