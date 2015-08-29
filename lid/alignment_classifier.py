from alignment_evaluation import alignment_features
import numpy as np
import nltk
from sklearn import linear_model
from sklearn.metrics import confusion_matrix, accuracy_score
import csv
import json
import argparse
import os
from database import ElasticConnection
import random
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.general_utils import alignment_tokenizer
from utils.general_utils import UnicodeWriter,UnicodeReader
import pickle
from sklearn.metrics import jaccard_similarity_score,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold



'''Contains code for both the features and model of the alignment classifier used to classify alignments as
    substantive or boiler-plate'''

def compute_tfidf_scores(alignment_data_path,pickle_file_name):
    count = 0
    alignment_docs = []
    for line in alignment_data_path:
        print count
        count += 1
        if count >= 100000:
            break
        json_obj = json.loads(line.strip())
        
        
        if "alignment_results" not in json_obj:
            continue

        for alignment_result in json_obj['alignment_results']:
            alignment_doc = []
            for section_alignment in alignment_result['alignments']:
                alignment_doc.extend([x for x in section_alignment['left'] if x not in ['-',None]])
                alignment_doc.extend([x for x in section_alignment['right'] if x not in ['-',None]])
            alignment_docs.append( " ".join(alignment_doc))
        
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(alignment_docs)
    idf = vectorizer.idf_
    idf = vectorizer._tfidf.idf_
    term_scores = zip(vectorizer.get_feature_names(), idf)
    term_dict = dict(term_scores)
    pickle_file = codecs.open(pickle_file_name,mode = "wb")
    pickle.dump(term_dict,pickle_file)
    return



def construct_training_set(alignments_file,out_file_name,score_threshold = None):
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
        print i
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
                if score < score_threshold:
                    continue
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



def evaluate_alignment_classifier():
    """runs k-fold cross validation on training set to evaluate classifier"""
    
    training_examples = []
    for line in csv.reader(self._training_file):
        if ( len(line[10].split()) != len(line[11].split()) ) or line[12] not in ["0","1"]:
            continue
        if len(line[10]) <= 1 or len(line[11]) < 1:
            continue
        training_examples.append({"left":line[10].split(),"right":line[11].split(),"label":int(line[12])})
    
    
    

    random.shuffle(training_examples)
    X,y = self.compute_feature_matrix(training_examples)

    self._model.fit(X_train,y_train)
    X,y = np.array(X),np.array(y)
    kf = KFold(n=len(X), n_folds=4, shuffle=False,
                           random_state=None)
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        self._model.fit(X_train,y_train)
        y_pred = self._model.predict(X_test)
        print classification_report(y_test, y_pred)
    
    self._model.fit(X,y)
    feat_names =  ['length','num_gaps_l','num_gaps_l','num_gaps_r','num_mismatches','num_matches','avg_gap_length_l',
            'avg_gap_length_r','avg_consec_match_length','jaccard_score','idf_mean','idf_medien']
    
    for x in zip(feat_names,self._model.coef_.tolist()):
        print x




class AlignmentClassifier():
    """Classifier that labels alignments as either substantive (1) or boilerplate (0)"""


    def __init__(self,idf_file_path):
        """Keyword Args:

            idf_file_path: file path of the table that stores idf scores of the words
            
        """
        self._idf_score_dict = pickle.load(open(idf_file_path))
        self._training_file = codecs.open(os.environ['POLICY_DIFFUSION']+\
                "/data/training_data_alignment_classifier_bigger.csv",mode = "rU")
        self._model =  LogisticRegression(penalty='l1')

    def compute_feature_matrix(self,training_examples):
        """Keywords Args:

            training_examples: list of dicts, where each dict contains alignment: "left":left_text,"right":right_text
                                and "label":label of alignment (1) substantive and boilerplate (0) 

            Returns:
                
                X: feature matrix
                y: labels

            """
        
        X = []
        y = []
        for training_example in training_examples:
            left = training_example['left']
            right = training_example['right']
            label = training_example['label']
            meta_features = self._compute_alignment_meta_features(left,right)
            idf_features  = self._compute_idf_score(left,right)
            features = meta_features + idf_features
            X.append(features)
            y.append(label)

	return X,y

    def train_model(self):
        """ Trains model using training examples in self._training_file and returns a trained model self._model
        
        Keywords Args:
            None

            Returns:
            None   

        """

        
        training_examples = []
        for line in csv.reader(self._training_file):
            if ( len(line[10].split()) != len(line[11].split()) ) or line[12] not in ["0","1"]:
                continue
            if len(line[10]) <= 1 or len(line[11]) < 1:
                continue
            training_examples.append({"left":line[10].split(),"right":line[11].split(),"label":int(line[12])})
        
        X,y = self.compute_feature_matrix(training_examples)

        self._model.fit(X_train,y_train)
        X,y = np.array(X),np.array(y)
        kf = KFold(n=len(X), n_folds=4, shuffle=False,
                               random_state=None)
        for train_index, test_index in kf:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self._model.fit(X_train,y_train)
            y_pred = self._model.predict(X_test)
            print classification_report(y_test, y_pred)
        
        self._model.fit(X,y)
        feat_names =  ['length','num_gaps_l','num_gaps_l','num_gaps_r','num_mismatches','num_matches','avg_gap_length_l',
                'avg_gap_length_r','avg_consec_match_length','jaccard_score','idf_mean','idf_medien']
        
        for x in zip(feat_names,self._model.coef_.tolist()):
            print x
                
    
    def predict(self,alignment_example):
        """predicts label for alignment example
        
        
        Keyword Args:

            alignment_example:  alignment [left,right] that needs to be labeled
        
        
        """
        pass

    
    
    def _compute_alignment_meta_features(self,left, right):
        '''
        This function takes as input two alignments and produce features of these
        '''
        #alignment features
        features = {}
        features['length'] = len(left)
        features['num_gaps_l'] = 0
        features['num_gaps_r'] = 0
        features['num_mismatches'] = 0
        features['num_matches'] = 0
        features['avg_gap_length_l'] = []
        features['avg_gap_length_r'] = []
        features['avg_consec_match_length'] = []
        features['jaccard_score'] = jaccard_similarity_score(left,right)

        #helper variables
        prev_gap_l = False
        prev_gap_r = False
        prev_match = False
        for i in range(len(left)):
            # print 'i: ', i
            # print 'features: ', features
            if left[i] == '-':
                features['num_gaps_l'] += 1
                if not prev_gap_l:
                    features['avg_gap_length_l'].append(1)
                    prev_gap_l = True
                else:
                    features['avg_gap_length_l'][-1] += 1
            else:
                prev_gap_l = False
            if right[i] == '-':
                features['num_gaps_r'] += 1
                if not prev_gap_r:
                    features['avg_gap_length_r'].append(1)
                    prev_gap_r = True
                else:
                    features['avg_gap_length_r'][-1] += 1
            else:
                prev_gap_r = False
            if left[i] != '-' and right[i] != '-':
                if left[i] != right[i]:
                    features['num_mismatches'] += 1
                elif left[i] == right[i]:
                    features['num_matches'] += 1
                    if not prev_match:
                        features['avg_consec_match_length'].append(1)
                        prev_match = True
                    else:
                        features['avg_consec_match_length'][-1] += 1
            if left[i] != right[i]:
                prev_match = False

        if features['avg_gap_length_l'] != []:
            features['avg_gap_length_l'] = np.mean(features['avg_gap_length_l'])
        else:
            features['avg_gap_length_l'] = 0
        if features['avg_gap_length_r'] != []:
            features['avg_gap_length_r'] = np.mean(features['avg_gap_length_r'])
        else:
            features['avg_gap_length_r'] = 0
        if features['avg_consec_match_length'] != []:
            features['avg_consec_match_length'] = np.mean(features['avg_consec_match_length'])
        else:
            features['avg_consec_match_length'] = 0

        features = sorted(features.items(),key = lambda x:x[0],reverse= False)
        return [x[1] for x in features]


    def _compute_idf_score(self,left,right):
        idf_scores = []

        for w in left:
            if w in self._idf_score_dict:
                idf_scores.append(self._idf_score_dict[w])

        for w in right:
            if w in self._idf_score_dict:
                idf_scores.append(self._idf_score_dict[w])
        

        return [np.mean(idf_scores),np.median(idf_scores)]
        
    


def main():
    parser = argparse.ArgumentParser(description='Classifier to label aligned text as "substantive" ')
    parser.add_argument('command',
            help='command to run, options are: construct_training_set,train_model,evaluate_model')
    parser.add_argument('--alignment_samples_doc', dest='alignment_samples',
            help="file path to the alignment samples used to construct training set ")

    args = parser.parse_args()

    if args.command == "construct_training_set":
        construct_training_set(open(args.alignment_samples),
                os.environ['POLICY_DIFFUSION']+"/data/classifier/alignments_training_set_high_scores.csv",50)    
    elif args.command == "compute_tfidf_scores":
        alignments_file = codecs.open("/mnt/data/sunlight/dssg/alignment_results/bill_to_bill_alignments.txt",
                encoding = "utf8")
        out_file = "/mnt/data/sunlight/dssg/features/alignment_tfidf_scores.p"
        compute_tfidf_scores(alignments_file,out_file)


    elif args.command == "train_model":
        pass
    elif args.command == "evaluate_model":
        pass
    else:
        print args
        print "command not recognized, please enter construct_training_set,train_model,evaluate_model"


if __name__ == "__main__":
    main()


