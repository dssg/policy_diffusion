from __future__ import division
from PyPDF2 import PdfFileReader
from bs4 import BeautifulSoup
import urllib2
#from docx import Document  # python-docx pacakage
import os
import codecs
import argparse
import ujson
import re
import base64
import textract
from tika import parser as tika_parser
try:
    from os import scandir, walk
except ImportError:
    from scandir import scandir, walk

################################
# global variables

TEST_PATH = "./DataExtractorTests/"
BILL_DATA_PATH = "/mnt/data/sunlight/dssg/scraped_bills"

################################
# helper functions

#loops through raw json files of state bills, extracts out bill text and creates json bills
def create_bill_documents(bill_data_path):

    bill_files = []
    for dirname, dirnames, filenames in walk(bill_data_path):
        for filename in filenames:
            bill_files.append(os.path.join(dirname, filename))

    outFile = codecs.open("/mnt/data/sunlight/dssg/extracted_data/extracted_bills.json",'w',encoding = "utf-8")
    for i,bill_file in enumerate(bill_files[260000:]):
        bill_json_obj = extract_bill_document(bill_file)
        outFile.write("{0}\n".format(ujson.encode(bill_json_obj)) )

        if i%100 == 0:
            print "{:.2f}% done".format(i/len(bill_files))



    outFile.close()



#helper function for extracting bill documents
def extract_bill_document(bill_file_path):
    data_dict = ujson.decode(open(bill_file_path).read())
    bill_dict = {}

    bill_text_count = [1 for x in data_dict['type'] if "bill" in x.lower()]
    if sum(bill_text_count) < 1:
        return None

    bill_id = re.sub("\s+", "", data_dict['bill_id'])

    try:
        if data_dict['versions'] == []:
            bill_document_first = ""
            bill_document_last = ""
        else:
            bill_document_first = base64.b64decode(data_dict['versions'][0]['bill_document'])
            bill_document_first = textract.process()
            bill_document_first = tika_parser.from_buffer(bill_document_first)['content']
            bill_document_last = base64.b64decode(data_dict['versions'][-1]['bill_document'])
            bill_document_last = tika_parser.from_buffer(bill_document_last)['content']

    except Exception as e:
        print e.args
        bill_document_first = None
        bill_document_last = None

    bill_dict['unique_id'] = "{0}_{1}_{2}".format(data_dict['state'], data_dict['session'], bill_id)
    bill_dict['bill_id'] = data_dict['bill_id']
    bill_dict['date_updated'] = data_dict['updated_at']
    bill_dict['session'] = data_dict['session']
    bill_dict['sunlight_id'] = data_dict['id']
    bill_dict['bill_title'] = data_dict['title']
    bill_dict['bill_type'] = data_dict['type']
    bill_dict['state'] = data_dict['state']
    bill_dict['chamber'] = data_dict['chamber']
    bill_dict['date_created'] = data_dict['created_at']
    bill_dict['bill_document_first'] = bill_document_first
    bill_dict['bill_document_last'] = bill_document_last
    bill_dict['actions'] = data_dict['actions']

    if "short_tite" in data_dict.keys():
        bill_dict['short_title'] = data_dict['short_title']
    elif "+short_title" in data_dict.keys():
        bill_dict['short_title'] = data_dict['+short_title']

    else:
        bill_dict['short_title'] = None

    if "summary" in data_dict.keys():
        bill_dict['summary'] = data_dict['summary']
    else:
        bill_dict['summary'] = None

    return bill_dict




def soupToText(soup):
    """
    Args:
        BeautifulSoup soup object of an html file

    Returns:
        a string of the text stripped of the javascript
    """

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()  # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text


################################
# pdf functions

def pdfToText(path):
    """
    Args:
        path to pdf file

    Returns:
        a string of the pdf
    """

    PDF = PdfFileReader(file(path, 'rb'))

    output = ""
    for page in PDF.pages:
        output = output + page.extractText()

    return output


################################
# html functions

def htmlToText(path):
    """
    Args:
        path to html file

    Returns:
        a string of the html file
    """
    with codecs.open(path, encoding='utf-8') as html:
        soup = BeautifulSoup(html)

    return soupToText(soup)


def urlToText(url):
    """
    Args:
        url

    Returns:
        a string of the text on the url
    """
    doc = urllib2.urlopen(url).read()
    soup = BeautifulSoup(doc)

    return soupToText(soup)


################################
# doc functions
# Good resource: http://www.konstantinkashin.com/blog/2013/09/25/scraping-pdfs-and-word-documents-with-python/

def docToText(path):
    """
    Turns docx file into a string.

    Note: it concatenates paragraphs and does not
    distinguish between beginning and end of paragraphs

    Args:
        path to docx file

    Returns:
        a string of the docx file
    """
    doc = Document(path)

    output = ""
    for paragraph in doc.paragraphs:
        output = output + paragraph.text

    return output


def oldDocToTest(path):
    """
    Turns doc file into a string.

    Args:
        path to doc file

    Returns:
        a string of the doc file
    """
    name = path[:-4]
    textFile = name + ".txt"
    os.system("antiword " + path + " > " + textFile)

    with codecs.open(textFile, encoding='utf-8') as text:
        output = text.read()
    os.system("rm " + textFile)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('command', help='command to run, options are: build_index')
    parser.add_argument('--data_path', dest='data_path', help="file path of data to be indexed ")

    args = parser.parse_args()

    if args.command == "extract_bills":
        create_bill_documents(args.data_path)
    elif args.command == "test_extractors":
        # Run tests
        print "pdf test: "
        print pdfToText(TEST_PATH + "test.pdf")

        print "urltest: "
        print urlToText('http://www.alec.org/model-legislation/72-hour-budget-review-act/')

        print "html test: "
        print htmlToText(TEST_PATH + "test.html")

        print "old doc test: "
        print oldDocToTest(TEST_PATH + 'test.doc')



