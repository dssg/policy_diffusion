import ujson
import base64
import urllib2
import socket
from ftplib import FTP, error_perm
import re
from StringIO import StringIO
import time
import multiprocessing


def alignment_tokenizer(s,type = "space"):
    if type == "space":
        s = s.split(" ")
    return s    

#creates a searalized json object for bill sources
def bill_source_to_json(url,source,date):
    jsonObj = {}
    jsonObj['url'] = url
    jsonObj['date'] = date
    jsonObj['source'] = base64.b64encode(source)

    return ujson.encode(jsonObj)

#creates a json object for bill sources (not encoded)
def bill_source_to_json_not_encoded(url,source,date):
    jsonObj = {}
    jsonObj['url'] = url
    jsonObj['date'] = date
    jsonObj['source'] = source

    return ujson.encode(jsonObj)

#wrapper for urllib2.urlopen that catches URLERROR and socket error
def fetch_url(url):

    #fetch ftp file
    if 'ftp://' in url:

        try:
            domain_pattern = re.compile("/[A-Za-z0-9\.]+")
            domain_name = domain_pattern.search(url).group(0)[1:]
            ftp = FTP(domain_name,timeout=10)
            ftp.login()
            file_name = "/".join(url.split("/")[3:])

            r = StringIO()
            ftp.retrbinary('RETR {0}'.format(file_name), r.write)
            document = r.getvalue()
            time.sleep(1)

        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            document = None


        return document

    #fetch http file
    else:

        try:
            req  = urllib2.urlopen(url,timeout=10)
            document = req.read()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            document = None

        return document

#used to find alignments in broader text
def find_subsequence(s,q):
    '''
    is the list s contained in q in order and if it is what are indices
    '''
    for i in range(len(q)):
        T = True
        for j in range(len(s)):
            if s[j] != q[i+j]:
                T = False
                break
        if T:
            return (i, i + j + 1)
    return (0,0)
