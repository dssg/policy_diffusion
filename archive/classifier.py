from alignment_evaluation import alignment_features
import numpy as np
import nltk
from sklearn import linear_model
from sklearn.metrics import confusion_matrix, accuracy_score

from score_alignments import StateTFIDF
import json
import argparse
import os
from database import ElasticConnection
import random
import codecs
from utils.general_utils import alignment_tokenizer
from utils.general_utils import UnicodeWriter
from sklearn.metrics import jaccard_similarity_score


def construct_training_set(alignments_file,out_file_name):
    """
    Args:
        alignments_file (file) -- file containing sample alignments
        
        out_file_name (string) -- name of training data file to write to

    Returns:
        None
    """
    ec = ElasticConnection(host= "54.203.12.145")
    
    training_examples = []
    for i,x in enumerate(alignments_file):
        json_obj = json.loads(x.strip())
        
        if "alignment_results" not in json_obj.keys():
            continue

        left_doc_id = json_obj['query_document_id']
        left_bill_title = ec.get_bill_by_id(left_doc_id)['bill_title']
        
        left_doc = json_obj['query_document']
        left_doc = reduce(lambda x,y:x+y,left_doc)
        
        left_doc_length = len(left_doc.split())

        for i,alignment_doc in enumerate(json_obj['alignment_results']):
            
            right_doc_id = alignment_doc['document_id']
            right_bill_title = ec.get_bill_by_id(right_doc_id)['bill_title']
            
            for alignment in alignment_doc['alignments']:

                left = alignment['left']
                right = alignment['right']
                left_start = alignment['left_start'] 
                right_start = alignment['right_start']
                left_end = alignment['left_end']
                right_end = alignment['right_end']
                score = alignment['score']
                training_examples.append([left_doc_id,right_doc_id,left_doc_length,left_start,right_start,left_end,
                    right_end,score,left_bill_title,right_bill_title,
                    " ".join(left)," ".join(right)])
        
    
    random.shuffle(training_examples)            
    
    header = ["left_doc_id","right_doc_id","left_doc_length","left_start","right_start","left_end",
                    "right_end","score","left_bill_title","right_bill_title","left","right"]
   

    k = 500
    with codecs.open(out_file_name, 'wb') as output_file:
        writer =  UnicodeWriter(output_file, header)
        writer.writerow(header)
        for l in training_examples[0:k]:
            l = [unicode(x) for x in l]
            writer.writerow(l)


    return
=======
from score_alignments import StateTFIDF
import json
import argparse
import os
from database import ElasticConnection
import random
import codecs
from utils.general_utils import alignment_tokenizer
from utils.general_utils import UnicodeWriter


def construct_training_set(alignments_file,out_file_name):
    """
    Args:
        alignments_file (file) -- file containing sample alignments
        
        out_file_name (string) -- name of training data file to write to

    Returns:
        None
    """
    ec = ElasticConnection(host= "54.203.12.145")
    
    training_examples = []
    for i,x in enumerate(alignments_file):
        json_obj = json.loads(x.strip())
        
        if "alignment_results" not in json_obj.keys():
            continue

        left_doc_id = json_obj['query_document_id']
        left_bill_title = ec.get_bill_by_id(left_doc_id)['bill_title']
        
        left_doc = json_obj['query_document']
        left_doc = reduce(lambda x,y:x+y,left_doc)
        
        left_doc_length = len(left_doc.split())

        for i,alignment_doc in enumerate(json_obj['alignment_results']):
            
            right_doc_id = alignment_doc['document_id']
            right_bill_title = ec.get_bill_by_id(right_doc_id)['bill_title']
            
            for alignment in alignment_doc['alignments']:

                left = alignment['left']
                right = alignment['right']
                left_start = alignment['left_start'] 
                right_start = alignment['right_start']
                left_end = alignment['left_end']
                right_end = alignment['right_end']
                score = alignment['score']
                training_examples.append([left_doc_id,right_doc_id,left_doc_length,left_start,right_start,left_end,
                    right_end,score,left_bill_title,right_bill_title,
                    " ".join(left)," ".join(right)])
        
    
    random.shuffle(training_examples)            
    
    header = ["left_doc_id","right_doc_id","left_doc_length","left_start","right_start","left_end",
                    "right_end","score","left_bill_title","right_bill_title","left","right"]
   

    k = 500
    with codecs.open(out_file_name, 'wb') as output_file:
        writer =  UnicodeWriter(output_file, header)
        writer.writerow(header)
        for l in training_examples[0:k]:
            l = [unicode(x) for x in l]
            writer.writerow(l)


    return


def features_matrix(alignment):
	right = alignment['right']
	left = alignment['left']
	features['left_tfidf'], features['right_tfidf'] = s.tfidf_score(left, right)
	features = alignment_features(left, right)
	features['score'] = alignment['score']
	features['label'] = alignment['label']

	return features

def evaluate_model():
    data = list_alignments
    featuresets = [features_matrix(alignment) for alignment in data]

    data_list = [[value['avg_consec_match_length'], value['avg_gap_length_l'], 
                            value['avg_gap_length_r'], value['jaccard_score'], 
                            value['length'], value['num_gaps_l'], value['num_gaps_r'], 
                            value['num_matches'], value['num_mismatches'], 
                            value['score'], value['label']] for value in featuresets]

    alignment_data = np.array(data_list)
    alignment_y=alignment_data[:,-1]
    alignment_X=alignment_data[:,:-1]

    # A random permutation, to split the data randomly
    np.random.seed(0)
    indices = np.random.permutation(len(alignment_X))
    train_n = 5
    alignment_X_train = alignment_X[indices[:-train_n]]
    alignment_y_train = alignment_y[indices[:-train_n]]
    alignment_X_test  = alignment_X[indices[-train_n:]]
    alignment_y_test  = alignment_y[indices[-train_n:]]
     
    # Create and fit a logistic regression
    logistic = linear_model.LogisticRegression(C=1e5)
    logistic.fit(alignment_X_train, alignment_y_train)
    y_pred = logistic.predict(alignment_X_test)

    #Calculate accuracy
    accuracy_score(alignment_y_test, y_pred)
    cm = confusion_matrix(alignment_y_test, y_pred)



def main():
    parser = argparse.ArgumentParser(description='Classifier to label aligned text as "substantive" ')
    parser.add_argument('command',
            help='command to run, options are: construct_training_set,train_model,evaluate_model')
    parser.add_argument('--alignment_samples_doc', dest='alignment_samples',
            help="file path to the alignment samples used to construct training set ")
    args = parser.parse_args()

    if args.command == "construct_training_set":
        construct_training_set(open(args.alignment_samples),
                os.environ['POLICY_DIFFUSION']+"/data/classifier/alignments_training_set.csv")    
    elif args.command == "train_model":
        pass
    elif args.command == "evaluate_model":
        pass
    else:
        print args
        print "command not recognized, please enter construct_training_set,train_model,evaluate_model"


if __name__ == "__main__":
    main()












