from SunlightTextUtils import pdfdata_to_text, text_after_line_numbers, worddata_to_text
import urllib2
import lxml.html
import re
import ujson

#helper functions
def check_extractor(doc_path, func, state):
	json_obj = ujson.decode(open(doc_path).read())
	url = "{0}/{1}".format("http://static.openstates.org/documents/" + state, json_obj['versions'][0]['doc_id'])
	doc = urllib2.urlopen(url).read()
	extracted = func(doc)
	print extracted

def check_extractor_doctypes(doc_path, func, state):
	json_obj = ujson.decode(open(doc_path).read())
	doc_type =json_obj['versions'][0]['mimetype']
	url = "{0}/{1}".format("http://static.openstates.org/documents/" + state, json_obj['versions'][0]['doc_id'])
	doc = urllib2.urlopen(url).read()
	extracted = func(doc_type, doc)
	print extracted

########
#tests

#nd

def nd_extract_text(doc_source):
    return text_after_line_numbers(pdfdata_to_text(doc_source))


path = "/mnt/data/sunlight/dssg/scraped_bills/nd/62/lower/HB_1462.json"
check_extractor(path, nd_extract_text, "nd")


#oh

def oh_extract_text(doc_source):
    doc = lxml.html.fromstring(doc_source)
    text = ' '.join(x.text_content() for x in doc.xpath('//td[@align="LEFT"]'))
    return text

path = '/mnt/data/sunlight/dssg/scraped_bills/oh/129/lower/HB_239.json'
check_extractor(path, oh_extract_text, "oh")


#ok

def ok_extract_text(doc_source):
    return worddata_to_text(doc_source)

path = '/mnt/data/sunlight/dssg/scraped_bills/ok/2011-2012/lower/HB_3100.json'
check_extractor(path, ok_extract_text, 'ok')


#or

def or_extract_text(doc_source):
    doc = lxml.html.fromstring(doc_source)
    lines = doc.xpath('//pre/text()')[0].splitlines()
    text = ' '.join(line for line in lines
                    if not re.findall('Page \d+$', line))
    return text

path = '/mnt/data/sunlight/dssg/scraped_bills/or/2011_Regular_Session/lower/HB_3652.json'
check_extractor(path, or_extract_text, 'or')


#pa

def pa_extract_text(doc, data):
    if doc in (None, 'text/html'):
        doc = lxml.html.fromstring(data)
        text = ' '.join(x.text_content() for x in doc.xpath('//tr/td[2]'))
        return text

path = "/mnt/data/sunlight/dssg/scraped_bills/pa/2009-2010/lower/HB_2711.json"
check_extractor_doctypes(path, pa_extract_text, 'pa')


#ri

def ri_extract_text(doc_source):
    return text_after_line_numbers(pdfdata_to_text(doc_source))

path = '/mnt/data/sunlight/dssg/scraped_bills/ri/2012/lower/HB_7540.json'
check_extractor(path, ri_extract_text, 'ri')


#sc

def sc_extract_text(data_source):
    doc = lxml.html.fromstring(data_source)
    # trim first and last part
    text = ' '.join(p.text_content() for p in doc.xpath('//p')[1:-1])
    return text

path = "/mnt/data/sunlight/dssg/scraped_bills/sc/2013-2014/lower/H_5154.json"
check_extractor(path, sc_extract_text, 'sc')


#sd

def sd_extract_text(data_source):
    doc = lxml.html.fromstring(data_source)
    return ' '.join(div.text_content() for div in
                    doc.xpath('//div[@align="full"]'))

path = '/mnt/data/sunlight/dssg/scraped_bills/sd/2010/lower/HC_1009.json'
check_extractor(path, sd_extract_text, 'sd')


#tn

def tn_extract_text(data_source):
    return ' '.join(line for line in pdfdata_to_text(data_source).splitlines()
                    if re.findall('[a-z]', line)).decode('utf8')

path = '/mnt/data/sunlight/dssg/scraped_bills/tn/107/lower/HJR_833.json'
check_extractor(path, tn_extract_text, 'tn')


#tx
def tx_extract_text(doc_source):
    doc = lxml.html.fromstring(doc_source)
    return doc.xpath('//html')[0].text_content()

path = '/mnt/data/sunlight/dssg/scraped_bills/tx/81/lower/HR_2889.json'
check_extractor(path, tx_extract_text, 'tx')

#ut
def ut_extract_text(doc_source):
    doc = lxml.html.fromstring(doc_source)
    return doc.xpath('//html')[0].text_content()

path = '/mnt/data/sunlight/dssg/scraped_bills/ut/2011/lower/HJR_5.json'
check_extractor(path, ut_extract_text, 'ut')

#vt
def vt_extract_text(doc_source):
    return text_after_line_numbers(pdfdata_to_text(doc_source))

path ='/mnt/data/sunlight/dssg/scraped_bills/vt/2009-2010/lower/H_741.json'
check_extractor(path, vt_extract_text, 'vt')

#va
def va_extract_text(doc_source):
    doc = lxml.html.fromstring(doc_source)
    text = ' '.join(x.text_content()
                    for x in doc.xpath('//div[@id="mainC"]/p'))
    return text

path = '/mnt/data/sunlight/dssg/scraped_bills/va/2010/lower/HB_823.json'
check_extractor(path, va_extract_text, 'va')

#wa
def wa_extract_text(doc_source):
    doc = lxml.html.fromstring(doc_source)
    text = ' '.join(x.text_content() for x in doc.xpath('//body/p'))
    return text

path = '/mnt/data/sunlight/dssg/scraped_bills/wa/2011-2012/lower/HB_2244.json'
check_extractor(path, wa_extract_text, 'wa')


#wv
def wv_extract_text(doc, data):
    if (doc.get('mimetype') == 'text/html' or 'bills_text.cfm' in doc['url']):
        doc = lxml.html.fromstring(data)
        return '\n'.join(p.text_content() for p in
                         doc.xpath('//div[@id="bhistcontent"]/p'))

path = '/mnt/data/sunlight/dssg/scraped_bills/wv/2011/lower/HCR_158.json'

#wi
def wi_extract_text(doc, data):
    is_pdf = (doc['mimetype'] == 'application/pdf' or
              doc['url'].endswith('.pdf'))
    if is_pdf:
        return text_after_line_numbers(pdfdata_to_text(data))

path = '/mnt/data/sunlight/dssg/scraped_bills/wi/2009_Regular_Session/lower/AB_91.json'


#wy
def wy_extract_text(doc_source):
    return ' '.join(line for line in pdfdata_to_text(doc_source).splitlines()
                    if re.findall('[a-z]', line))

path = '/mnt/data/sunlight/dssg/scraped_bills/wy/2011/lower/HB_83.json'
check_extractor(path, wy_extract_text, 'wy')

#dc
def dc_extract_text(doc_source):
    lines = pdfdata_to_text(doc_source).splitlines()
    no_big_indent = re.compile('^\s{0,10}\S')
    text = '\n'.join(line for line in lines if no_big_indent.match(line))
    return text

