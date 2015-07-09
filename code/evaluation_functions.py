import urllib2
from tika import parser

   #Evaluation Functions

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
         'http://www.legis.state.wv.us/Bill_Status/bills_text.cfm?billdoc=hb2874%20intr.htm&yr=2015&sesstype=RS&i=2874'],
         ['ftp://ftp.cga.ct.gov/2015/tob/h/2015HB-06784-R00-HB.htm',
         'http://www.capitol.hawaii.gov/session2015/bills/SB129_.PDF'],
         ['http://nebraskalegislature.gov/FloorDocs/104/PDF/Intro/LB493.pdf',
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



def create_matches(ls):
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
        for url in urls[2:]:
            print "bill_id: " + str(bill_id)
            bills[bill_id] = {}
            doc = urllib2.urlopen(url).read()

            try:
              text = parser.from_buffer(doc)['content']
              print text
            except UnicodeDecodeError:
              print 'did not extract correctly'
              bill_id +=1
              continue 
            bills[bill_id]['url'] = url
            bills[bill_id]['text'] = text
            bills[bill_id]['match'] = k
            bill_id += 1

        k += 1
    return matches

