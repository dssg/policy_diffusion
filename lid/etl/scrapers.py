import json
import codecs
import base64
import logging
import re
import os
import sys
import multiprocessing
import utils
import random
import argparse
import traceback
import urllib2
from config import DATA_PATH
from bs4 import BeautifulSoup


try:
    from os import scandir, walk
except ImportError:
    from scandir import scandir, walk

BILL_SCRAPER_LOG = os.environ['POLICY_DIFFUSION'] + '/logs/bill_scraper.log'


# scrapes all bills from the input data path
def scrape_all_bills(bill_data_path, num_workers):
    logging.basicConfig(filename=BILL_SCRAPER_LOG, level=logging.DEBUG)

    bill_file_paths = []
    for dirname, dirnames, filenames in walk(bill_data_path):
        for filename in filenames:
            bill_file_paths.append(os.path.join(dirname, filename))


    scrape_bill_document_from_sunlight(bill_file_paths[0])

    random.shuffle(bill_file_paths)

    pool = multiprocessing.Pool(num_workers)
    
    print "fetch {0} urls from sunlight...".format(len(bill_file_paths))
    pool.map(scrape_bill_document_from_sunlight, bill_file_paths)

    print "finished fetching urls..."


# open individual json file and scrape bill document,
# from the s3 server provided by sunlight foundation
def scrape_bill_document_from_sunlight(file_path):
    try:
        file_path = file_path.strip()

        #define path to write file
        out_file_path = file_path.split("/bills")[-1]
        out_file_path = re.sub("\s+", "_", out_file_path)
        out_dir_root_path = "{0}/scraped_bills".format(DATA_PATH)
        out_file_name = "{0}{1}.json".format(out_dir_root_path, out_file_path)

        bill_json = json.loads(codecs.open(file_path, encoding="utf8").read())

        # filter versions to be only the first and last
        try:
            bill_json['versions'] = [bill_json['versions'][0], bill_json['versions'][-1]]
        except IndexError:
            return

        base_url = "{0}/{1}".format("http://static.openstates.org/documents", bill_json['state'])
        urls = ["{0}/{1}".format(base_url, x['doc_id']) for x in bill_json['versions']]
        source_urls = [x['url'] for x in bill_json['versions']]

        for i, url in enumerate(urls):

            bill_document = utils.fetch_url(url)

            #hash bill using base64
            if bill_document is not None:
                bill_document = base64.b64encode(bill_document)
            else:
                logging.error("file {0}, url {1}, version {2}, error: << {3} >>".format(
                    file_path, url, i, "link error"))

            bill_json['versions'][i]['bill_document'] = bill_document

        if not os.path.exists(os.path.dirname(out_file_name)):
            os.makedirs(os.path.dirname(out_file_name))
        with codecs.open(out_file_name, "w", encoding="utf8") as f:
            f.write(json.dumps(bill_json))

        logging.info("successfully scraped bill: {0}".format(out_file_path))

    except Exception as e:
        trace_message = re.sub("\n+", "\t", traceback.format_exc())
        trace_message = re.sub("\s+", " ", trace_message)
        trace_message = "<<{0}>>".format(trace_message)
        m = "Failed to obtain documents for {0}: {1}".format(file_path, trace_message)
        logging.error(m)

    return


# scrapes bill document from original source link
# this is a backup if s3 doesn't work
def scrape_bill_document_from_original_source(filePath):
    filePath = filePath.strip()

    outFilePath = "/".join(filePath.split("/")[7:])
    outFilePath = re.sub("\s+", "_", outFilePath)
    outDirRootPath = "/mnt/data/sunlight/dssg/scraped_bills_new"
    outFileName = "{0}/{1}.json".format(outDirRootPath, outFilePath)

    billFile = codecs.open(filePath, encoding="utf8").read()
    billJson = json.loads(billFile)

    # filters documents that are resolutions
    bill_text_count = [1 for x in billJson['type'] if "bill" in x.lower()]
    if sum(bill_text_count) < 1:
        return

    # filter versions to be only the first and last
    billJson['versions'] = [billJson['versions'][0], billJson['versions'][-1]]

    urls = [x['url'] for x in billJson['versions']]

    for i, url in enumerate(urls):

        billDocument = utils.fetch_url(url)

        if billDocument is not None:
            billDocument = base64.b64encode(billDocument)
        else:
            logging.error("file {0}, url {1}, version {2}, error: << {3} >>".format(filePath, url, i, "link error"))

        billJson['versions'][i]['bill_document'] = billDocument

    if not os.path.exists(os.path.dirname(outFileName)):
        os.makedirs(os.path.dirname(outFileName))
    with codecs.open(outFileName, "w", encoding="utf8") as f:
        f.write(json.dumps(billJson))

    logging.info("successfully scraped bill: {0}".format(outFilePath))

    return


# scrapes model legistlation from ALEC's official site
# and the tracker website ALEC exposed
def scrape_ALEC_model_legislation():
    url = 'http://www.alec.org/model-legislation/'
    response = urllib2.urlopen(url).read()
    bs = BeautifulSoup(response, 'html5')

    # Get all links from website
    ALEClist = []
    for link in bs.find_all('a'):
        if link.has_attr('href'):
            ALEClist.append(link.attrs['href'])

    # Filter list so that we have only the ones with model-legislation
    ALEClinks = []
    i = 0
    for i in range(0, len(ALEClist)):
        if ALEClist[i][20:38] == "model-legislation/":
            ALEClinks.append(ALEClist[i])
            i = i + 1

    # To get only unique links (get rid off duplicates)
    ALEClinks = set(ALEClinks)

    # Save to json file
    with open('{0}/data/model_legislation/alec_bills.json'.format(DATA_PATH, 'w')) as f:
        for line in ALEClinks:
            source = urllib2.urlopen(line).read()
            url = line
            date = 2015
            Jsonbill = bill_source_to_json(url, source, date)
            f.write("{0}\n".format(Jsonbill))

    # Save old alec bills (from Center for the Media and Democracy)
def scrape_alec_exposed_bills ():
    names = os.listdir('{0}/model_legislation/ALEC_exposed'.format(DATA_PATH))
    with open('alec_old_bills.json', 'w') as f2:
    for name in names:
        try:
            text = tp.from_file(name)
            source = text['content']
        except:
            source = None
        url = None
        date = '2010-2013'
        print name
        print source
        Jsonbill = bill_source_to_json_not_encoded(url, source, date)
        f2.write("{0}\n".format(Jsonbill))


def scrape_CSG_model_legislation():
url = 'http://www.csg.org/programs/policyprograms/SSL.aspx'
doc = urllib2.urlopen(url).read()
bs = BeautifulSoup(doc)

links = []
for link in bs.find_all('a'):
    if link.has_attr('href'):
        candidate = link.attrs['href']
        # links with pdf extension tend to be model bills
        if candidate[-4:] == ".pdf":
            links.append(candidate)

# only keeps distinct links
links2 = list(set(links))

badCount = 0
goodCount = 0

with open('csg_bills.json', 'w') as f:
    for line in links2:
        try:
            url_key = {}
            source = urllib2.urlopen(line).read()
            Jsonbill = bill_source_to_json(link, source, None)
            f.write("{0}\n".format(Jsonbill))
            goodCount += 1
        except:
            badCount += 1
    print line

print str(badCount) + " did not work"


def scrape_ALICE_legislation():
    path = "/mnt/data/sunlight/dssg/model_legislation/links_"
    lines = []
    for i in [1, 2, 3]:
        filePath = path + str(i) + ".txt"
        with open(filePath) as f:
            lines.extend(f.read().splitlines())

    text = ''.join(lines)
    bs = BeautifulSoup(text)

    links = []
    for link in bs.find_all('a'):
        if link.has_attr('href'):
            links.append(link.attrs['href'])


    # grab pdfs from links
    billList = []
    for url in links:
        doc = urllib2.urlopen(url).read()
        bs = BeautifulSoup(doc)

        for link in bs.find_all('a'):
            if link.has_attr('href'):
                candidate = link.attrs['href']
                if candidate[-4:] == ".pdf":  # links with pdf extension tend to be model bills
                    billList.append("https://stateinnovation.org" + candidate)

    badCount = 0
    goodCount = 0
    with open('alice_bills.json', 'w') as f:
        for link in billList:
            # url_key = {}
            # source = urllib2.urlopen(link).read()
            # Jsonbill = bill_source_to_json(link, source, None)
            # f.write("{0}\n".format(Jsonbill))
            try:
                source = urllib2.urlopen(link).read()
                Jsonbill = bill_source_to_json(link, source, None)
                f.write("{0}\n".format(Jsonbill))
                goodCount += 1
            except:
                badCount += 1

    print str(badCount) + " did not work"

def scrape_misc_legislation():
        # Access list of clean urls
with open('/mnt/data/sunlight/dssg/model_legislation/clean_urls.txt',
    'r') as f:
links = f.read().splitlines()

badCount = 0
goodCount = 0
with open('misc_bills.json', 'w') as jsonfile:
    for link in links:
        try:
            source = urllib2.urlopen(link).read()
            Jsonbill = bill_source_to_json(link, source, None)
            jsonfile.write("{0}\n".format(Jsonbill))
            goodCount += 1
            print goodCount
        except:
            badCount += 1

    print str(badCount) + " did not work"
    print str(goodCount) + " worked"



def main():

    parser = argparse.ArgumentParser(description='module that contains functions to scrape legislative data\ '
                                                 ' from sunlight foundation and various'
                                                 'lobbying organizations')
    parser.add_argument('command', help='command to run, options are: \n scrape_bills_from_sunlight')
    parser.add_argument('--data_path', dest='data_path', help="file path of data to be indexed ")
    parser.add_argument('--num_workers', dest='num_workers',default = 10,
                        type = int, help="file path of data to be indexed ")

    args = parser.parse_args()

    if args.command == "scrape_bills_from_sunlight":
        scrape_all_bills(args.data_path,args.num_workers)
    elif args.command == "scrape_ALEC_legislation":
        scrape_ALEC_model_legislation()
    elif args.command == "scrape_CSG_legislation":
        scrape_CSG_model_legislation()
    elif args.command == "scrape_ALICE_legislation":
        scrape_ALICE_legislation()
    elif args.command =="scrape_misc_legislation":
        scrape_misc_legislation()
    else:
        print("command not recognized, use -h flag to see list available commands")



if __name__ == "__main__":
    main()
