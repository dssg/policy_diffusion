import lxml.html
import urllib2

#Louisiana (LA)


def extract_text_la(doc_source):
    return text_after_line_numbers(pdfdata_to_text(doc_source))

url_la = 'http://www.legis.la.gov/Legis/ViewDocument.aspx?d=958299'
doc_source_la = urllib2.urlopen(url_la).read()
test_la = extract_text_la(doc_source_la)


#Maine (ME)

def extract_text_me(doc_source):
    doc = lxml.html.fromstring(doc_source)
    return doc.xpath('//div[@class="billtextbody"]')[0].text_content()

url_me = 'http://www.mainelegislature.org/legis/bills/bills_127th/billtexts/HP098801.asp'
doc_source_me = urllib2.urlopen(url_me).read()
test_me = extract_text_me(doc_source_me)

#Maryland (MD)

def extract_text_md(doc_source):
    text = pdfdata_to_text(doc_source)
    return text_after_line_numbers(text)

url_md = 'http://mgaleg.maryland.gov/2015RS/bills/sb/sb0270t.pdf'
doc_source_md = urllib2.urlopen(url_md).read()
test_md = extract_text_md(doc_source_md)

#Massachusetts (MA)

def extract_text_ma(doc_source):
    doc = lxml.html.fromstring(doc_source)
    text = ' '.join([x.text_content()
                     for x in doc.xpath('//td[@class="longTextContent"]//p')])
    return text

url_ma =  'https://s3.amazonaws.com/documents.openstates.org/MAD00001112'
doc_source_ma = urllib2.urlopen(url_ma).read()
test_ma = extract_text_ma(doc_source_ma)


#Michigan (MI)

def extract_text_mi(doc_source):
    doc = lxml.html.fromstring(doc_source)
    text = doc.xpath('//body')[0].text_content()
    return text

url_mi = 'http://www.legislature.mi.gov/documents/2015-2016/billintroduced/Senate/htm/2015-SIB-0403.htm'
doc_source_mi = urllib2.urlopen(url_mi).read()
test_mi = extract_text_mi(doc_source_mi)


#Mississippi (MS)
def extract_text_ms(doc_source):
    doc = lxml.html.fromstring(doc_source)
    text = ' '.join(p.text_content() for p in
                    doc.xpath('//h2/following-sibling::p'))
    return text

url_ms = 'http://billstatus.ls.state.ms.us/documents/2015/html/SB/2100-2199/SB2161SG.htm'
doc_source_ms = urllib2.urlopen(url_ms).read()
test_ms = extract_text_ms(doc_source_ms)

#Missouri (MO)
def extract_text_mo(doc_source):
    text = pdfdata_to_text(doc_source)
    return text_after_line_numbers(text).encode('ascii', 'ignore')

url_mo = 'http://house.mo.gov/billtracking/bills151/billpdf/perf/HJR0044P.PDF'
doc_source_mo = urllib2.urlopen(url_mo).read()
test_mo = extract_text_mo(doc_source_mo)

#Montana (MT)

def extract_text_mt(doc_source):
    return text_after_line_numbers(pdfdata_to_text(doc_source))

url_mt = http://static.openstates.org/documents/mt/MTD00005634
doc_source_mt = urllib2.urlopen(url_mt).read()
test_mt = extract_text_mt(doc_source_mt)

#Nebraska (NE)

def extract_text_ne(doc_source):
    text = pdfdata_to_text(doc_source)
    return text
url_ne = 'http://static.openstates.org/documents/ne/NED00050004'
doc_source_ne = urllib2.urlopen(url_ne).read()
test_ne = extract_text_ne(doc_source_ne)

#Nevada (NV)

def extract_text_nv(doc_source):
    return text_after_line_numbers(pdfdata_to_text(doc_source))

url_nv = 'http://www.leg.state.nv.us/Session/78th2015/Bills/SB/SB515.pdf'
doc_source_nv = urllib2.urlopen(url_nv).read()
test_nv = extract_text_nv(doc_source_nv)

#New Hampshire (NH)

def extract_text_nh(doc_source):
    doc = lxml.html.fromstring(doc_source)
    return doc.xpath('//html')[0].text_content()

url_nh = 'http://www.gencourt.state.nh.us/legislation/2011/HB0546.html'
doc_source_nh =  urllib2.urlopen(url_nh).read()
test_nh = extract_text_nh(doc_source_nh)

#New Jersey (NJ)

def extract_text_nj(doc_source):
    doc = lxml.html.fromstring(doc_source)
    text = doc.xpath('//div[@class="Section3"]')[0].text_content()
    return text

url_nj = 'http://www.njleg.state.nj.us/2014/Bills/AJR/120_I1.HTM'
doc_source_nj = urllib2.urlopen(url_nj).read()
test_nj = extract_text_nj(doc_source_nj)

#New Mexico(NM)

def extract_text_nm(doc_source):
    doc = lxml.html.fromstring(doc_source)
    return doc.xpath('//body')[0].text_content().split(
        u'\r\n\xa0\r\n\xa0\r\n\xa0')[-1]

url_nm = 'http://static.openstates.org/documents/nm/NMD00006762'
doc_source_nm = urllib2.urlopen(url_nm).read()
test_nm = extract_text_nm(doc_source_nm)

#New York(NY)

def extract_text_ny(doc_source):
    doc = lxml.html.fromstring(doc_source)
    text = doc.xpath('//pre')[0].text_content()
    # if there's a header above a _________, ditch it
    text = text.rsplit('__________', 1)[-1]
    # strip numbers from lines (not all lines have numbers though)
    text = re.sub('\n\s*\d+\s*', ' ', text)
    return text

url_ny = 'http://static.openstates.org/documents/ny/NYD00186972'
doc_source_ny = urllib2.urlopen(url_ny).read()
test_ny = extract_text_ny(doc_source_ny)

#North Carolina (NC)

def extract_text_nc(doc_source):
    doc = lxml.html.fromstring(doc_source)
    text = ' '.join([x.text_content() for x in
                     doc.xpath('//p[starts-with(@class, "a")]')])
    return text

url_nc = 'http://static.openstates.org/documents/nc/NCD00003673'
doc_source_nc =  urllib2.urlopen(url_nc).read()
test_nc = extract_text_nc(doc_source_nc)


#Idaho(ID)

def extract_text_id(doc_source):
    return text_after_line_numbers(pdfdata_to_text(doc_source))

url_id = 'http://static.openstates.org/documents/id/IDD00000717'
doc_source_id = urllib2.urlopen(url_id).read()
test_id = extract_text_id(doc_source_id)

#Illinois (IL)

def extract_text_il(doc_source):
    doc = lxml.html.fromstring(doc_source)
    text = ' '.join(x.text_content() for x in doc.xpath('//td[@class="xsl"]'))
    return text

url_il = 'http://static.openstates.org/documents/il/ILD00012923'
doc_source_il = urllib2.urlopen(url_il ).read()
test_il = extract_text_il(doc_source_il)

#Indiana

def extract_text_in(doc_source):
    text = pdfdata_to_text(doc_source)
    return text_after_line_numbers(text)

url_in = 'http://static.openstates.org/documents/in/IND00000313'
doc_source_in = urllib2.urlopen(url_in).read()
test_in = extract_text_in(doc_source_in)

#Iowa (IA)

def extract_text_ia(doc_source):
    doc = lxml.html.fromstring(doc_source)
    text = doc.xpath('//pre')[0].text_content()
    # strip two sets of line numbers
    return text_after_line_numbers(text_after_line_numbers(text))

url_ia = 'http://static.openstates.org/documents/ia/IAD00005333'
doc_source_ia = urllib2.urlopen(url_ia).read()
test_ia =  extract_text_ia(doc_source_ia)

#Kansas(KS)

def extract_text_ks(doc_source):
    return text_after_line_numbers(pdfdata_to_text(doc_source))

url_ks = 'http://static.openstates.org/documents/ks/KSD00001620'
doc_source_ks = urllib2.urlopen(url_ks).read()
test_ks = extract_text_ks(doc_source_ks)

#Kentucky (KY)

def extract_text_ky(doc_source):
    return worddata_to_text(doc_source)

url_ky = 'http://static.openstates.org/documents/ky/KYD00002532'
doc_source_ky = urllib2.urlopen(url_ky).read()
test_ky = extract_text_ky(doc_source_ky)