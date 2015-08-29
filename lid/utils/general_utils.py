import ujson
import base64
import urllib2
import socket
from ftplib import FTP, error_perm
import re
from StringIO import StringIO
import time
import multiprocessing
import pickle
import multiprocessing as mp
import gc
import signal
import csv
import codecs
import cStringIO

#######Code from http://www.filosophy.org/post/32/python_function_execution_deadlines__in_simple_examples/ #########

class TimedOutExc(Exception):
    pass

def deadline(timeout, *args):

    def decorate(f):
        def handler(signum, frame):
            raise TimedOutExc()
        
        def new_f(*args):

            signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout)
            return f(*args)
            signa.alarm(0)

        new_f.__name__ = f.__name__
        return new_f
    return decorate

#######Code from http://www.filosophy.org/post/32/python_function_execution_deadlines__in_simple_examples/ #########

class UTF8Recoder:
    """
    Iterator that reads an encoded stream and reencodes the input to UTF-8
    """
    def __init__(self, f, encoding):
        self.reader = codecs.getreader(encoding)(f)

    def __iter__(self):
        return self

    def next(self):
        return self.reader.next().encode("utf-8")


class UnicodeReader():
    """
    A CSV reader which will iterate over lines in the CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        f = UTF8Recoder(f, encoding)
        self.reader = csv.reader(f, dialect=dialect, **kwds)

    def next(self):
        row = self.reader.next()
        return [unicode(s, "utf-8") for s in row]

    def __iter__(self):
        return self




class UnicodeWriter():
    def __init__(self, f, dialect=csv.excel, encoding="utf-8-sig", **kwds):
        self.queue = cStringIO.StringIO()
        self.writer = csv.writer(self.queue, dialect=dialect, **kwds)
        self.stream = f
        self.encoder = codecs.getincrementalencoder(encoding)()
    def writerow(self, row):
        '''writerow(unicode) -> None
        This function takes a Unicode string and encodes it to the output.
        '''
        self.writer.writerow([s.encode("utf-8") for s in row])
        data = self.queue.getvalue()
        data = data.decode("utf-8")
        data = self.encoder.encode(data)
        self.stream.write(data)
        self.queue.truncate(0)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)

#********DEPRECATED*************
class WorkerPool():

    def __init__(self,num_workers=1,worker_timeout = 600):
        
        self._num_workers = num_workers
        self._worker_timeout = worker_timeout
        self._results = mp.Queue()
        self._pool = [None]*self._num_workers
        self._worker_times = [0.0]*self._num_workers

    def _assign_new_task(self,worker_id,input_args):
        p = self._pool[worker_id]
        p.join()
        arg = input_args.pop()
        new_p = mp.Process(target= func,args = (arg,self._results),name = ('process_'+str(worker_id)))
        new_p.start()
        self._pool[worker_id] = new_p
        self._worker_times[worker_id] = time.time()
        
    def work(self,func,input_args):
        worker_counter = 0
        #define wrapper function that queues result from input func
        def new_func(x):
            y = func(*x)
            self._results.put(y)

        
        while len(input_args) > 0 or ("running" in status):
            
            #assign new worker tasks to empty pool slots
            for i in range(self._num_workers):
                    
                if len(input_args) > 0 and self._pool[i] is None:
                    arg = input_args.pop(0)
                    new_p = mp.Process(target= new_func,args = (arg,),name = ('process_'+str(i)))
                    new_p.start()
                    print worker_counter
                    worker_counter+=1
                    self._pool[i] = new_p
                    self._worker_times[i] = time.time()

            time.sleep(0.1)
            status = self.check_pool_status(time.time())
            import numpy as np
            print time.time() - np.array(self._worker_times)
            for i in range(len(status)):
                if status[i] == "completed":
                    p = self._pool[i]
                    p.terminate()
                    p.join()
                    self._pool[i] = None
                    del p
                elif status[i] == "timeout":
                    p = self._pool[i]
                    p.terminate()
                    self._pool[i] = None
                    print "terminated job  ",p.name
                    gc.collect()
        
        result_list = []

        while not self._results.empty():
            result_list.append( self._results.get() )

        return result_list

    #returns a list of bools indicating running status of each worker. 
    #running,timeout,completed
    def check_pool_status(self,current_time):
        status_list = []
        for i in range(self._num_workers):

            worker = self._pool[i]
            if worker is None:
                status_list.append("closed")
            elif worker.is_alive() and (current_time-self._worker_times[i]<self._worker_timeout):
                status_list.append("running")
            elif worker.is_alive() and (current_time-self._worker_times[i]>=self._worker_timeout):
                status_list.append("timeout")
            elif not worker.is_alive():
                status_list.append("completed")

        return status_list
# ********DEPRECATED*************


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


def load_pickle(name):
    with open('{0}.p'.format(name),'rb') as fp:
        f =pickle.load(fp)

    return f


def save_pickle(thing, name):
    with open('{0}.p'.format(name),'wb') as fp:
        pickle.dump(thing, fp)
