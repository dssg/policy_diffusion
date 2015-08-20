from alignment_evaluation import alignment_features
import numpy as np
import nltk
from sklearn import linear_model
from sklearn.metrics import confusion_matrix, accuracy_score
import csv
from score_alignment import StateTFIDF

#create alignment list training set
alignment_list = []
i = 0
for alignment_results in alignments:
	if 'alignment_results' in alignment_results:
		for alignment in alignment_results['alignment_results']:
			alignment_list.extend(alignment['alignments'])

for i in range(len(alignment_list)):
	alignment_list[i]['left'] = ' '.join(alignment_list[i]['left']).encode('utf-8')
	alignment_list[i]['right'] = ' '.join(alignment_list[i]['right']).encode('utf-8')


		
alignment_to_csv = alignment_list[0:100]
with open('training_set.csv', 'wb') as output_file:
    keys = alignment_list[0].keys()
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(alignment_to_csv)




#for i in range(len(list_alignments)):
#	list_alignments[i]['label'] = np.random.binomial(1,.5)


def features_matrix(alignment):
	right = alignment['right']
	left = alignment['left']
	features = alignment_features(left, right)
	features['score'] = alignment['score']
	features['label'] = alignment['label']

	return features

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


######################
