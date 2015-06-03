import ujson
import codecs
import base64
import urllib2
from pprint import pprint
import logging
import time

LOG_FILENAME = 'bill_scraper.log'
logging.basicConfig(filename=LOG_FILENAME,
                    level=logging.DEBUG,
                    )


outDirRootPath = "/mnt/data/sunlight/dssg/scraped_bills/"


filePaths = open("/mnt/data/sunlight/dssg/bill_file_paths.txt").readlines()

badBills = 0
for filePath in filePaths:
    time.sleep(0.33)
    filePath = filePath.strip()
    outFilePath = "/".join(filePath.split("/")[7:])

    billFile = codecs.open( filePath,encoding = "utf8" ).read()
    billJson = ujson.decode(billFile)

    try:
        url = billJson['versions'][-1]["url"]

    except:
        badBills += 1
        logging.error("file: {0} , error: {1}".format(filePath,"no link error"))
        continue

    try:
        billDocument = urllib2.urlopen(url).read()
    except  URLError:
        badBills+=1
        logging.error("file: {0} , error: {1}".format(filePath,"broke url error"))

    billJson['bill_document'] = billDocument
    outFile = open("{0}/{1}".format(outDirRootPath,outFilePath),'w')
    outFile.write(ujson.encode(billJson))
    exit()


pprint (fileTypeCounts)
