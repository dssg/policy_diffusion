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
        out_file_path = "/".join(file_path.split("/")[6:])
        out_file_path = re.sub("\s+", "_", out_file_path)
        out_dir_root_path = "/mnt/data/sunlight/dssg/scraped_bills_new"
        out_file_name = "{0}/{1}.json".format(out_dir_root_path, out_file_path)

        bill_json = json.loads(codecs.open(file_path, encoding="utf8").read())

        # filters documents that are resolutions
        bill_text_count = [1 for x in bill_json['type'] if "bill" in x.lower()]
        if sum(bill_text_count) < 1:
            return


        # filter versions to be only the first and last
        try:
            bill_json['versions'] = [bill_json['versions'][0], bill_json['versions'][-1]]
        except IndexError:
            return

        base_url = "{0}/{1}".format("http://static.openstates.org/documents", bill_json['state'])
        urls = ["{0}/{1}".format(base_url, x['doc_id']) for x in bill_json['versions']]

        for i, url in enumerate(urls):

            bill_document = utils.fetch_url(url)

            #hash bill using base64
            if bill_document is not None:
                bill_document = base64.b64encode(bill_document)
            else:
                logging.error("file {0}, url {1}, version {2}, error: << {3} >>".format(file_path, url, i, "link error"))

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
    else:
        print("command not recognized, use -h flag to see list available commands")



if __name__ == "__main__":
    main()
