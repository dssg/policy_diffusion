from __future__ import division
import lxml.html
import urllib2
from utils.sunlight_utils import pdfdata_to_text, text_after_line_numbers, worddata_to_text
import urllib2
import ujson
import lxml.html
import gc
import base64
import re
import random
import os
import traceback
from pprint import pprint

try:
    from os import scandir, walk
except ImportError:
    from scandir import scandir, walk


###################################
# Text extractors for every state #
###################################
def al_text_extractor(doc_source):
    text = pdfdata_to_text(doc_source)
    return text_after_line_numbers(text)


def ak_text_extractor(doc_source):
    doc = lxml.html.fromstring(doc_source)
    text = doc.xpath('//pre')[0].text_content()
    text = text_after_line_numbers(text)
    return text


def ar_text_extractor(doc_source):
    return text_after_line_numbers(pdfdata_to_text(doc_source))


def az_text_extractor(mimetype, doc_source):
    if mimetype == 'text/html':
        doc = lxml.html.fromstring(doc_source)
        text = doc.xpath('//div[@class="Section2"]')[0].text_content()
        return text
    else:
        return text_after_line_numbers(pdfdata_to_text(doc_source))


def ca_text_extractor(doc_source):
    doc = lxml.html.fromstring(doc_source)
    divs_to_try = ['//div[@id="bill"]', '//div[@id="bill_all"]']
    for xpath in divs_to_try:
        div = doc.xpath(xpath)
        if div:
            return div[0].text_content()


def co_text_extractor(doc_source):
    return worddata_to_text(doc_source)


def ct_text_extractor(doc_source):
    doc = lxml.html.fromstring(doc_source)
    text = ' '.join(p.text_content() for p in doc.xpath('//body/p'))
    return text


def de_text_extractor(mimetype, doc_source):
    if mimetype == 'text/html':
        doc = lxml.html.fromstring(doc_source)
        return ' '.join(x.text_content()
                        for x in doc.xpath('//p[@class="MsoNormal"]'))


def fl_text_extractor(doc_source):
    doc = lxml.html.fromstring(doc_source)
    pre = doc.xpath('//pre')
    if pre:
        text = pre[0].text_content().encode('ascii', 'replace')
        return text_after_line_numbers(text)
    else:
        return '\n'.join(x.text_content() for x in doc.xpath('//tr/td[2]'))


def ga_text_extractor(doc_source):
    doc = lxml.html.fromstring(doc_source)
    lines = doc.xpath('//span/text()')
    headers = ('A\r\nRESOLUTION', 'AN\r\nACT')
    # take off everything before one of the headers
    for header in headers:
        if header in lines:
            text = '\n'.join(lines[lines.index(header) + 1:])
            break
    else:
        text = ' '.join(lines)

    return text


def hi_text_extractor(mimetype, doc_source):
    if mimetype == 'application/pdf':
        return text_after_line_numbers(pdfdata_to_text(doc_source))
    else:
        return None


def la_text_extractor(doc_source):
    return text_after_line_numbers(pdfdata_to_text(doc_source))


def me_text_extractor(doc_source):
    doc = lxml.html.fromstring(doc_source)
    return doc.xpath('//div[@class="billtextbody"]')[0].text_content()


def md_text_extractor(doc_source):
    text = pdfdata_to_text(doc_source)
    return text_after_line_numbers(text)


def ma_text_extractor(doc_source):
    doc = lxml.html.fromstring(doc_source)
    text = ' '.join([x.text_content()
                     for x in doc.xpath('//td[@class="longTextContent"]//p')])
    return text


def mi_text_extractor(doc_source):
    doc = lxml.html.fromstring(doc_source)
    text = doc.xpath('//body')[0].text_content()
    return text


def ms_text_extractor(doc_source):
    doc = lxml.html.fromstring(doc_source)
    text = ' '.join(p.text_content() for p in
                    doc.xpath('//h2/following-sibling::p'))
    return text


def mo_text_extractor(doc_source):
    text = pdfdata_to_text(doc_source)
    return text_after_line_numbers(text).encode('ascii', 'ignore')

def mn_text_extractor(mimetype, doc_source):
    doc = lxml.html.fromstring(doc_source)
    xtend = doc.xpath('//div[@class="xtend"]')[0].text_content()
    for v in doc.xpath('.//var/text()'):
        xtend = xtend.replace(v, '')
    doc = None
    gc.collect()
    return xtend

def mt_text_extractor(doc_source):
    return text_after_line_numbers(pdfdata_to_text(doc_source))


def ne_text_extractor(doc_source):
    text = pdfdata_to_text(doc_source)
    return text


def nv_text_extractor(doc_source):
    return text_after_line_numbers(pdfdata_to_text(doc_source))


def nh_text_extractor(doc_source):
    doc = lxml.html.fromstring(doc_source)
    return doc.xpath('//html')[0].text_content()


def nj_text_extractor(doc_source):
    doc = lxml.html.fromstring(doc_source)
    text = doc.xpath('//div[@class="Section3"]')[0].text_content()
    return text


def nm_text_extractor(doc_source):
    doc = lxml.html.fromstring(doc_source)
    return doc.xpath('//body')[0].text_content().split(
        u'\r\n\xa0\r\n\xa0\r\n\xa0')[-1]


def ny_text_extractor(doc_source):
    doc = lxml.html.fromstring(doc_source)
    text = doc.xpath('//pre')[0].text_content()
    # if there's a header above a _________, ditch it
    text = text.rsplit('__________', 1)[-1]
    # strip numbers from lines (not all lines have numbers though)
    text = re.sub('\n\s*\d+\s*', ' ', text)
    return text


def nc_text_extractor(doc_source):
    doc = lxml.html.fromstring(doc_source)
    text = ' '.join([x.text_content() for x in
                     doc.xpath('//p[starts-with(@class, "a")]')])
    return text




def nd_text_extractor(doc_source):
    return text_after_line_numbers(pdfdata_to_text(doc_source))



def oh_text_extractor(doc_source):
    doc = lxml.html.fromstring(doc_source)
    text = ' '.join(x.text_content() for x in doc.xpath('//td[@align="LEFT"]'))
    return text


def ok_text_extractor(doc_source):
    return worddata_to_text(doc_source)


def or_text_extractor(doc_source):
    doc = lxml.html.fromstring(doc_source)
    lines = doc.xpath('//pre/text()')[0].splitlines()
    text = ' '.join(line for line in lines
                    if not re.findall('Page \d+$', line))
    return text


def pa_text_extractor(mimetype, data):
    if mimetype in (None, 'text/html'):
        doc = lxml.html.fromstring(data)
        text = ' '.join(x.text_content() for x in doc.xpath('//tr/td[2]'))
        return text

def ri_text_extractor(doc_source):
    return text_after_line_numbers(pdfdata_to_text(doc_source))


def sc_text_extractor(data_source):
    doc = lxml.html.fromstring(data_source)
    # trim first and last part
    text = ' '.join(p.text_content() for p in doc.xpath('//p')[1:-1])
    return text

def sd_text_extractor(data_source):
    doc = lxml.html.fromstring(data_source)
    return ' '.join(div.text_content() for div in
                    doc.xpath('//div[@align="full"]'))


def tn_text_extractor(data_source):
    return ' '.join(line for line in pdfdata_to_text(data_source).splitlines()
                    if re.findall('[a-z]', line)).decode('utf8')


def tx_text_extractor(doc_source):
    doc = lxml.html.fromstring(doc_source)
    return doc.xpath('//html')[0].text_content()


def ut_text_extractor(mimetype, data):
    if mimetype == 'application/pdf':
        return text_after_line_numbers(pdfdata_to_text(data))


def vt_text_extractor(doc_source):
    return text_after_line_numbers(pdfdata_to_text(doc_source))


def va_text_extractor(doc_source):
    doc = lxml.html.fromstring(doc_source)
    text = ' '.join(x.text_content()
                    for x in doc.xpath('//div[@id="mainC"]/p'))
    return text


def wa_text_extractor(doc_source):
    doc = lxml.html.fromstring(doc_source)
    text = ' '.join(x.text_content() for x in doc.xpath('//body/p'))
    return text


def wv_text_extractor(mimetype, url, doc_source):
    if (mimetype == 'text/html' or 'bills_text.cfm' in url):
        doc = lxml.html.fromstring(doc_source)
        return '\n'.join(p.text_content() for p in
                         doc.xpath('//div[@id="bhistcontent"]/p'))


def wi_text_extractor(mimetype, url, data):
    is_pdf = (mimetype == 'application/pdf' or
              url.endswith('.pdf'))
    if is_pdf:
        return text_after_line_numbers(pdfdata_to_text(data))


def wy_text_extractor(doc_source):
    return ' '.join(line for line in pdfdata_to_text(doc_source).splitlines()
                    if re.findall('[a-z]', line))


def id_text_extractor(doc_source):
    return text_after_line_numbers(pdfdata_to_text(doc_source))


def il_text_extractor(doc_source):
    doc = lxml.html.fromstring(doc_source)
    text = ' '.join(x.text_content() for x in doc.xpath('//td[@class="xsl"]'))
    return text



def in_text_extractor(doc_source):
    text = pdfdata_to_text(doc_source)
    return text_after_line_numbers(text)


def ia_text_extractor(doc_source):
    doc = lxml.html.fromstring(doc_source)
    text = doc.xpath('//pre')[0].text_content()
    # strip two sets of line numbers
    return text_after_line_numbers(text_after_line_numbers(text))


def ks_text_extractor(doc_source):
    return text_after_line_numbers(pdfdata_to_text(doc_source))


def ky_text_extractor(doc_source):
    return worddata_to_text(doc_source)

def dc_text_extractor(doc_source):
    lines = pdfdata_to_text(doc_source).splitlines()
    no_big_indent = re.compile('^\s{0,10}\S')
    text = '\n'.join(line for line in lines if no_big_indent.match(line))
    return text

def pr_text_extractor(doc_source):
    return worddata_to_text(doc_source)



# meta function that calls the individual state extractors
def bill_text_extractor(state_name, doc_source, mimetype,url):

    try:
        if state_name == "al":
            bill_text = al_text_extractor(doc_source)

        elif state_name == "ak":
            bill_text = ak_text_extractor(doc_source)

        elif state_name == "ar":
            bill_text = ar_text_extractor(doc_source)

        elif state_name == "az":
            bill_text = az_text_extractor(mimetype,doc_source)

        elif state_name == "ca":
            bill_text = ca_text_extractor(doc_source)

        elif state_name == "co":
            bill_text = co_text_extractor(doc_source)

        elif state_name == "ct":
            bill_text = ct_text_extractor(doc_source)

        elif state_name == "de":
            bill_text = de_text_extractor(mimetype,doc_source)

        elif state_name == "fl":
            bill_text = fl_text_extractor(doc_source)

        elif state_name == "ga":
            bill_text = ga_text_extractor(doc_source)

        elif state_name == "hi":
            bill_text = hi_text_extractor(mimetype,doc_source)

        elif state_name == "la":
            bill_text = la_text_extractor(doc_source)

        elif state_name == "me":
            bill_text = me_text_extractor(doc_source)

        elif state_name == "md":
            bill_text = md_text_extractor(doc_source)

        elif state_name == "ma":
            bill_text = ma_text_extractor(doc_source)

        elif state_name == "mi":
            bill_text = mi_text_extractor(doc_source)

        elif state_name == "ms":
            bill_text = ms_text_extractor(doc_source)

        elif state_name == "mo":
            bill_text = mo_text_extractor(doc_source)

        elif state_name == "mt":
            bill_text = mt_text_extractor(doc_source)

        elif state_name == "ne":
            bill_text = ne_text_extractor(doc_source)

        elif state_name == "nv":
            bill_text = nv_text_extractor(doc_source)

        elif state_name == "nh":
            bill_text = nh_text_extractor(doc_source)

        elif state_name == "nm":
            bill_text = nm_text_extractor(doc_source)

        elif state_name == "nj":
            bill_text = nj_text_extractor(doc_source)

        elif state_name == "ny":
            bill_text = ny_text_extractor(doc_source)

        elif state_name == "nc":
            bill_text = nc_text_extractor(doc_source)

        elif state_name == "nd":
            bill_text = nd_text_extractor(doc_source)

        elif state_name == "oh":
            bill_text = oh_text_extractor(doc_source)

        elif state_name == "ok":
            bill_text = ok_text_extractor(doc_source)

        elif state_name == "or":
            bill_text = or_text_extractor(doc_source)

        elif state_name == "pa":
            bill_text = pa_text_extractor(doc_source)

        elif state_name == "ri":
            bill_text = ri_text_extractor(doc_source)

        elif state_name == "sc":
            bill_text = sc_text_extractor(doc_source)

        elif state_name == "sd":
            bill_text = sd_text_extractor(doc_source)

        elif state_name == "tn":
            bill_text = tn_text_extractor(doc_source)

        elif state_name == "tx":
            bill_text = tx_text_extractor(doc_source)

        elif state_name == "ut":
            bill_text = ut_text_extractor(doc_source)

        elif state_name == "vt":
            bill_text = vt_text_extractor(doc_source)

        elif state_name == "va":
            bill_text = va_text_extractor(doc_source)

        elif state_name == "wa":
            bill_text = wa_text_extractor(doc_source)

        elif state_name == "wv":
            bill_text = wv_text_extractor(mimetype,url,doc_source)

        elif state_name == "wi":
            bill_text = wi_text_extractor(mimetype,url,doc_source)

        elif state_name == "wy":
            bill_text = wy_text_extractor(doc_source)

        elif state_name == "id":
            bill_text = id_text_extractor(doc_source)

        elif state_name == "il":
            bill_text = il_text_extractor(doc_source)

        elif state_name == "in":
            bill_text = in_text_extractor(doc_source)

        elif state_name == "ia":
            bill_text = ia_text_extractor(doc_source)

        elif state_name == "ks":
            bill_text = ks_text_extractor(doc_source)

        elif state_name == "ky":
            bill_text = ky_text_extractor(doc_source)

        elif state_name == "dc":
            bill_text = dc_text_extractor(doc_source)

        elif state_name == "pr":
            bill_text = pr_text_extractor(doc_source)

        elif state_name == "mn":
            bill_text = mn_text_extractor(mimetype, doc_source)

        if len(bill_text) < 100:
            return None
        else:
            return bill_text

    except Exception as e:
        return None



def test_bill_extractors():

    base_path = "/mnt/data/sunlight/dssg/scraped_bills/"
    state_codes = os.listdir("/mnt/data/sunlight/dssg/scraped_bills/")


    for state_code in state_codes:
        data_path = "{0}/{1}".format(base_path,state_code)
        bill_files = []
        for dirname, dirnames, filenames in walk(data_path):
            for filename in filenames:
                bill_files.append(os.path.join(dirname, filename))

        random.shuffle(bill_files)

        num_tests = 10
        num_errors = 0
        for i,bill_file in enumerate(bill_files[0:num_tests]):
            json_obj = ujson.decode(open(bill_file).read())


            try:
                bill_document = base64.b64decode(json_obj['versions'][0]['bill_document'])
            except:
                num_tests -= 1
                continue


            try:
                mimetype = json_obj['versions'][0]['mimetype']
            except KeyError:
                mimetype = json_obj['versions'][0]['+mimetype']


            bill_text = bill_text_extractor(state_code,bill_document,mimetype,json_obj['versions'][0]['url'])

            if bill_text == None:
                num_errors +=1


        if 100*(1-(num_errors/num_tests)) < 100.0:

            output =  "passed {:.2f}% number of tests for state {:s} with {:d} tests".format(
                100*(1-(num_errors/num_tests)),state_code,num_tests)
            print output.upper()




def main():
    print "running test of bills extractors for each state"
    test_bill_extractors()



if __name__ == "__main__":
    main()
