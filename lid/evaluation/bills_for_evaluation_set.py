from elasticsearch import Elasticsearch
import re
import csv
import urllib2
import urllib
from urllib import urlopen
from tika import parser
import pickle


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
        for url,state in urls:
            try:
                print "bill_id: " + str(bill_id)
                bills[bill_id] = {}
                doc = urllib2.urlopen(url).read()
                text = parser.from_buffer(doc)['content'] 
                bills[bill_id]['url'] = url
                bills[bill_id]['text'] = text
                bills[bill_id]['match'] = k
                bills[bill_id]['state'] = state
            except:
                pass
                bad_count += 1
                print 'bad_count: ', bad_count
            bill_id += 1
        k += 1

    #get more evaluation bills
    eval_bills = grab_more_eval_bills()
    for more_bills in eval_bills:
        print 'bill_group: ' k
        k +=1
        for text, state in more_bills:
            bill_id += 1
            print 'bill_id: ', i

            bills[bill_id] = {}
            bills[bill_id]['text'] = text
            bills[bill_id]['state'] = state  
            bills[bill_id]['match'] = k

    try:
        for bill in bills.keys():
            if bills[bill] == {} or bills[bill]['text'] == '' \
                or bills[bill]['text'] == None:
                
                del bills[bill]
    except:
        pass

    return bills

def get_bill_by_id(unique_id):
    es = Elasticsearch(['54.203.12.145:9200', '54.203.12.145:9200'], timeout=300)
    match = es.search(index="state_bills", body={"query": {"match": {'unique_id': unique_id}}})
    bill_text = match['hits']['hits'][0]['_source']['bill_document_first']
    return bill_text

def grab_more_eval_bills():
    with open('../../data/evaluation_set/bills_for_evaluation_set.csv') as f:
        bills_list = [row for row in csv.reader(f.read().splitlines())]
        
    bill_ids_list = []
    url_lists = []
    topic_list = []
    for i in range(len(bills_list)):
        state = bills_list[i][1]
        if state == 'ct':
            continue
        topic = bills_list[i][0]
        bill_number = bills_list[i][2]
        bill_number = re.sub(' ', '', bill_number)
        year = bills_list[i][3]
        url = bills_list[i][6]
        unique_id = str(state + '_' + year + '_' + bill_number)
        topic_list.append(topic)
        bill_ids_list.append(unique_id)
        url_lists.append(url)

    bills_ids = zip(bill_ids_list, url_lists)

    bad_count = 0
    bills_text = []
    state_list = []
    for i in range(len(bills_ids)):
        try:
            bill_text = get_bill_by_id(bills_ids[i][0])
        except IndexError:
            try:
                url = bills_ids[i][1]
                doc = urllib.urlopen(url).read()
                bill_text = parser.from_buffer(doc)['content']
                print url
            except IOError:
            	bad_count += 1 
            	print 'bad_count: ', bad_count
            	#skip this case
                continue
        bills_text.append(bill_text)
        state = bills_ids[i][0][0:2]
        state_list.append(state)

    bills_state = zip(bills_text, state_list, topic_list)

    bill_type_1 = []
    bill_type_2 = []
    for bill in bills_state:
        if bill[-1] == 'Adult Guardianship and Protective Proceedings Jurisdiction Act':
            bill_type_1.append((bill[0],bill[1]))
        else:
            bill_type_2.append((bill[0],bill[1]))

    return [bill_type_2, bill_type_1]

def create_save_bills(bill_list):
    bills = create_bills(bill_list)
    with open('../../data/evaluation_set/labeled_bills.p', 'wb') as fp:
        pickle.dump(bills, fp)

    return bills


if __name__ == '__main__':
	    #each list in this list of lists contains bills that are matches
    similar_bills = [[('http://www.azleg.gov/legtext/52leg/1r/bills/hb2505p.pdf', 'az'),
    ('http://www.legis.state.ak.us/basis/get_bill_text.asp?hsid=SB0012B&session=29', 'ak' ),
    ('http://www.capitol.hawaii.gov/session2015/bills/HB9_.PDF', 'hi'),
    ('http://www.capitol.hawaii.gov/session2015/bills/HB1047_.PDF', 'hi'),
    ('http://flsenate.gov/Session/Bill/2015/1490/BillText/Filed/HTML','fl'),
    ('http://ilga.gov/legislation/fulltext.asp?DocName=09900SB1836&GA=99&SessionId=88&DocTypeId=SB&LegID=88673&DocNum=1836&GAID=13&Session=&print=true','il'),
    ('http://www.legis.la.gov/Legis/ViewDocument.aspx?d=933306', 'la'),
    ('http://mgaleg.maryland.gov/2015RS/bills/sb/sb0040f.pdf', 'md'),
    ('http://www.legislature.mi.gov/documents/2015-2016/billintroduced/House/htm/2015-HIB-4167.htm', 'mi'),
    ('https://www.revisor.mn.gov/bills/text.php?number=HF549&version=0&session=ls89&session_year=2015&session_number=0','mn'),
    ('http://www.njleg.state.nj.us/2014/Bills/A2500/2354_R2.HTM','nj'),
    ('http://assembly.state.ny.us/leg/?sh=printbill&bn=A735&term=2015','ny'),
    ('http://www.ncga.state.nc.us/Sessions/2015/Bills/House/HTML/H270v1.html','nc'),
    ('https://olis.leg.state.or.us/liz/2015R1/Downloads/MeasureDocument/HB2005/A-Engrossed','or'),
    ('https://olis.leg.state.or.us/liz/2015R1/Downloads/MeasureDocument/SB947/Introduced','or'),
    ('http://www.legis.state.pa.us/CFDOCS/Legis/PN/Public/btCheck.cfm?txtType=HTM&sessYr=2015&sessInd=0&billBody=H&billTyp=B&billNbr=0624&pn=0724', 'pa'),
    ('http://www.scstatehouse.gov/sess121_2015-2016/prever/172_20141203.htm','sc'),
    ('http://lawfilesext.leg.wa.gov/Biennium/2015-16/Htm/Bills/House%20Bills/1356.htm', 'wa'),
    ('http://www.legis.state.wv.us/Bill_Status/bills_text.cfm?billdoc=hb2874%20intr.htm&yr=2015&sesstype=RS&i=2874','wv'),
    ('http://www.legis.state.wv.us/Bill_Status/bills_text.cfm?billdoc=hb2874%20intr.htm&yr=2015&sesstype=RS&i=2874', 'wv'),
    # ('ftp://ftp.cga.ct.gov/2015/tob/h/2015HB-06784-R00-HB.htm','ct'),
    ('http://www.capitol.hawaii.gov/session2015/bills/SB129_.PDF','hi'),
    ('http://nebraskalegislature.gov/FloorDocs/104/PDF/Intro/LB493.pdf', 'ne'),
    ('http://www.gencourt.state.nh.us/legislation/2015/HB0600.html', 'nh')],
    [('http://alecexposed.org/w/images/2/2d/7K5-No_Sanctuary_Cities_for_Illegal_Immigrants_Act_Exposed.pdf', 'model_legislation'),
    ('http://www.kslegislature.org/li_2012/b2011_12/measures/documents/hb2578_00_0000.pdf', 'ks'),
    ('http://flsenate.gov/Session/Bill/2011/0237/BillText/Filed/HTML','fl'),
    ('http://openstates.org/al/bills/2012rs/SB211/','al'),
    ('http://le.utah.gov/~2011/bills/static/HB0497.html','ut'),
    ('http://webserver1.lsb.state.ok.us/cf_pdf/2013-14%20FLR/HFLR/HB1436%20HFLR.PDF','ok')],
    [('http://www.alec.org/model-legislation/the-disclosure-of-hydraulic-fracturing-fluid-composition-act/', 'model_legislation'),
    ('ftp://ftp.legis.state.tx.us/bills/82R/billtext/html/house_bills/HB03300_HB03399/HB03328S.htm', 'tx')],
    [('http://www.legislature.mi.gov/(S(ntrjry55mpj5pv55bv1wd155))/documents/2005-2006/billintroduced/House/htm/2005-HIB-5153.htm', 'mi'),
    ('http://www.schouse.gov/sess116_2005-2006/bills/4301.htm','sc'),
    ('http://www.lrc.ky.gov/record/06rs/SB38.htm', 'ky'),
    ('http://www.okhouse.gov/Legislation/BillFiles/hb2615cs%20db.PDF', 'ok'),
    ('http://state.tn.us/sos/acts/105/pub/pc0210.pdf', 'tn'),
    ('https://docs.legis.wisconsin.gov/2011/related/proposals/ab69', 'wi'),
    ('http://legisweb.state.wy.us/2008/Enroll/HB0137.pdf', 'wy'),
    ('http://www.kansas.gov/government/legislative/bills/2006/366.pdf', 'ks'),
    ('http://billstatus.ls.state.ms.us/documents/2006/pdf/SB/2400-2499/SB2426SG.pdf', 'mi')],
    [('http://www.alec.org/model-legislation/state-withdrawal-from-regional-climate-initiatives/', 'model_legislation'),
    ('http://www.legislature.mi.gov/documents/2011-2012/resolutionintroduced/House/htm/2011-HIR-0134.htm', 'mi'),
    ('http://www.nmlegis.gov/Sessions/11%20Regular/memorials/house/HJM024.html', 'nm')],
    [('http://alecexposed.org/w/images/9/90/7J1-Campus_Personal_Protection_Act_Exposed.pdf', 'model_legislation'),
    ('ftp://ftp.legis.state.tx.us/bills/831/billtext/html/house_bills/HB00001_HB00099/HB00056I.htm', 'tx')],
    # [
    # ('http://essexuu.org/ctstat.html', 'ct'), we don't have connecituc
    # ('http://alisondb.legislature.state.al.us/alison/codeofalabama/constitution/1901/CA-170364.htm', 'al')],
    [('http://www.legis.state.ak.us/basis/get_bill_text.asp?hsid=HB0162A&session=27', 'ak'),
    ('https://legiscan.com/AL/text/HB19/id/327641/Alabama-2011-HB19-Enrolled.pdf', 'al'),
    ('http://www.leg.state.co.us/clics/clics2012a/csl.nsf/fsbillcont3/0039C9417C9D9D5D87257981007F3CC9?open&file=1111_01.pdf', 'co'),
    ('http://www.capitol.hawaii.gov/session2012/Bills/HB2221_.PDF', 'hi'),
    ('http://ilga.gov/legislation/fulltext.asp?DocName=09700HB3058&GA=97&SessionId=84&DocTypeId=HB&LegID=60409&DocNum=3058&GAID=11&Session=&print=true', 'il'),
    ('http://coolice.legis.iowa.gov/Legislation/84thGA/Bills/SenateFiles/Introduced/SF142.html', 'ia'),
    ('ftp://www.arkleg.state.ar.us/Bills/2011/Public/HB1797.pdf','ar'),
    ('http://billstatus.ls.state.ms.us/documents/2012/html/HB/0900-0999/HB0921SG.htm', 'ms'),
    ('http://www.leg.state.nv.us/Session/76th2011/Bills/SB/SB373.pdf', 'nv'),
    ('http://www.njleg.state.nj.us/2012/Bills/A1000/674_I1.HTM', 'nj'),
    ('http://webserver1.lsb.state.ok.us/cf_pdf/2011-12%20INT/hB/HB2821%20INT.PDF', 'ok'),
    ('http://www.legis.state.pa.us/CFDOCS/Legis/PN/Public/btCheck.cfm?txtType=PDF&sessYr=2011&sessInd=0&billBody=H&billTyp=B&billNbr=0934&pn=1003', 'pa'),
    ('http://www.capitol.tn.gov/Bills/107/Bill/SB0016.pdf', 'tn')],
    [('http://www.legislature.idaho.gov/idstat/Title39/T39CH6SECT39-608.htm', 'id'),
    ('http://www.legis.nd.gov/cencode/t12-1c20.pdf?20150708171557', 'nd')]
    ]

    bills = create_save_bills(similar_bills)




		


