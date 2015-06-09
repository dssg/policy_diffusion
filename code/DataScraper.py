import ujson
import json
import codecs
import base64
import urllib2
import logging
import time
import re
import os
import sys
import multiprocessing
import traceback

BILL_SCRAPER_LOG = os.environ['POLICY_DIFFUSION'] + '/logs/bill_scraper.log'



# scrapes all bills from file in /misc/bill_file_paths_random.tx
def scrape_all_bills():
    logging.basicConfig(filename=BILL_SCRAPER_LOG, level=logging.DEBUG)

    filePaths = open("/mnt/data/sunlight/dssg/misc/bill_file_paths_random.txt").readlines()

    pool = multiprocessing.Pool(20)

    pool.map(scrape_bill_document_from_sunlight, filePaths)


#    for filePath in filePaths[0:100]:
#        scrape_bill_document(filePath,outDirRootPath)

# open individual json file and scrape bill document,
# from the original source, usually state website
def scrape_bill_document_from_original_source(filePath):
    try:
        filePath = filePath.strip()

        outFilePath = "/".join(filePath.split("/")[7:])
        outFilePath = re.sub("\s+", "_", outFilePath)
        outDirRootPath = "/mnt/data/sunlight/dssg/scraped_bills"
        outFileName = "{0}/{1}.json".format(outDirRootPath, outFilePath)

        # check if file exists
        if os.path.isfile(outFileName):
            return

        billFile = codecs.open(filePath, encoding="utf8").read()
        billJson = ujson.decode(billFile)

        urls = [x['url'] for x in billJson['versions']]
        urls = [re.sub("\s", "%20", url) for url in urls]
        for i, url in enumerate(urls):

            time.sleep(1.0)
            scrapeCount = 0
            while scrapeCount <= 3:

                try:
                    url = billJson['versions'][-1]["url"]
                    url = re.sub("\s", "%20", url)
                    billDocument = urllib2.urlopen(url, timeout=10).read()
                    billDocument = base64.b64encode(billDocument)
                    billJson['versions'][i]['bill_document'] = billDocument
                    break
                except urllib2.URLError:
                    billJson['versions'][i]['bill_document'] = None
                    logging.error("file {0} ,version {1}, error: << {2} >>".format(filePath, i, "no link error"))
                    break

            if scrapeCount > 3:
                billJson['versions'][i]['bill_document'] = None
                logging.error("file: {0} , error: {1}, url:  {2}".format(filePath, "broke url error", url))
                break

        if not os.path.exists(os.path.dirname(outFileName)):
            os.makedirs(os.path.dirname(outFileName))
        with codecs.open(outFileName, "w", encoding="utf8") as f:
            f.write(json.dumps(billJson))

        logging.info("successfully scraped bill: {0}".format(outFilePath))

        return

    except Exception as e:
        traceMessage = re.sub("\n+", "\t", traceback.format_exc())
        traceMessage = re.sub("\s+", " ", traceMessage)
        traceMessage = "<<{0}>>".format(traceMessage)
        m = "Failed to obtain documents for {0}: {1}".format(filePath, traceMessage)
        logging.error(m)


# open individual json file and scrape bill document,
# from the s3 server provided by sunlight foundation
def scrape_bill_document_from_sunlight(filePath):
    try:
        filePath = filePath.strip()

        outFilePath = "/".join(filePath.split("/")[7:])
        outFilePath = re.sub("\s+", "_", outFilePath)
        outDirRootPath = "/mnt/data/sunlight/dssg/scraped_bills"
        outFileName = "{0}/{1}.json".format(outDirRootPath, outFilePath)

        billFile = codecs.open(filePath, encoding="utf8").read()
        billJson = ujson.decode(billFile)

        baseUrl = "{0}/{1}".format("http://static.openstates.org/documents", filePath.split("/")[7])
        urls = ["{0}/{1}".format(baseUrl, x['doc_id']) for x in billJson['versions']]

        for i, url in enumerate(urls):

            try:
                billDocument = urllib2.urlopen(url, timeout=10).read()
                billDocument = base64.b64encode(billDocument)
                billJson['versions'][i]['bill_document'] = billDocument
            except urllib2.URLError:
                billJson['versions'][i]['bill_document'] = None
                logging.error("file {0} ,version {1}, error: << {2} >>".format(filePath, i, "link error"))

        if not os.path.exists(os.path.dirname(outFileName)):
            os.makedirs(os.path.dirname(outFileName))
        with codecs.open(outFileName, "w", encoding="utf8") as f:
            f.write(json.dumps(billJson))

        logging.info("successfully scraped bill: {0}".format(outFilePath))

        return

    except Exception as e:
        traceMessage = re.sub("\n+", "\t", traceback.format_exc())
        traceMessage = re.sub("\s+", " ", traceMessage)
        traceMessage = "<<{0}>>".format(traceMessage)
        m = "Failed to obtain documents for {0}: {1}".format(filePath, traceMessage)
        logging.error(m)


def main():
    command = sys.argv[1]

    if command == "scrape_bills":
        scrape_all_bills()


if __name__ == "__main__":
    main()
