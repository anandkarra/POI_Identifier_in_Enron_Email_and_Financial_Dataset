import sys
import pickle
import math
import pprint
from time import time
from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.metrics import accuracy_score, classification_report


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                'salary',
                'bonus',
                'exercised_stock_options',
                'total_stock_value',
                'from_poi_to_this_person',
                'from_this_person_to_poi',
                'expenses',
                'total_payments'] # You will need to use more features

"""
Only these three features will be used in the following sections for implementing the
POI identifier.

For full details about the feature selection, consult feature_selection.py and
new_feature_performance.py.
"""

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL',0) # Full outlier exploration in identify_outliers.py

### Task 3: Create new feature(s)
for key,value in data_dict.items():

    from_poi_to_this_person_isnan = math.isnan(float(value['from_poi_to_this_person']))
    from_this_person_to_poi_isnan = math.isnan(float(value['from_this_person_to_poi']))

    if((not from_poi_to_this_person_isnan) and (not from_this_person_to_poi_isnan)):
        percent_from_poi_to_this_person = value['from_poi_to_this_person']/value['from_messages']
        percent_from_this_person_to_poi = value['from_this_person_to_poi']/value['to_messages']

        value['percent_from_poi_to_this_person'] = percent_from_poi_to_this_person
        value['percent_from_this_person_to_poi'] = percent_from_this_person_to_poi

    else:
        value['percent_from_poi_to_this_person'] = 'NaN'
        value['percent_from_this_person_to_poi'] = 'NaN'

### Feature scaling
def featureScaling(val,max_val,min_val):
    val = float(val)
    max_val = float(max_val)
    min_val = float(min_val)
    if min_val != max_val:
        scaled = (val-min_val)/(max_val-min_val)
        return scaled

for feature in features_list:
    if feature not in ['poi','percent_from_this_person_to_poi','percent_from_poi_to_this_person']:
        first = True

        for key,value in data_dict.items():
            if not math.isnan(float(value[feature])):
                if first:
                    max_val = value[feature]
                    min_val = value[feature]
                    first = False
                else:
                    if value[feature] > max_val:
                        max_val = value[feature]
                    if value[feature] < min_val:
                        min_val = value[feature]
    
        for key,value in data_dict.items():
            if not math.isnan(float(value[feature])):
                value[feature] = (featureScaling(value[feature],max_val,min_val))

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

features_train = features[:75]
features_test = features[75:]

labels_train = labels[:75]
labels_test = labels[75:]
#features_train,features_test,labels_train,labels_test = \
    #cross_validation.train_test_split(features,labels,test_size=0.3,random_state=42)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

# SVC
print "----- SVC -----"
from sklearn.svm import SVC
clf = SVC()

t0 = time()
clf.fit(features_train,labels_train)
print "Training time: ",round(time()-t0,3),"s"


t1= time()
pred = clf.predict(features_test)
print("Predicting time: ",round(time()-t1,3),"s")

print "Accuracy= ", round(clf.score(features_test,labels_test),2)

print classification_report(labels_test,pred)

# Decision Tree
print "----- Decision Tree -----"
from sklearn import tree
clf = tree.DecisionTreeClassifier()

t0 = time()
clf.fit(features_train,labels_train)
print "Training time: ",round(time()-t0,3),"s"


t1= time()
pred = clf.predict(features_test)
print("Predicting time: ",round(time()-t1,3),"s")

print "Accuracy= ", round(clf.score(features_test,labels_test),2)

print classification_report(labels_test,pred)

"""
The SVC gives a slightly higher accuracy than the decision tree classifier but 
overall the decision tree classifier performs better (considering precision, recall,
f1-score).
"""

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

print "---------------------------------------------"
print "----- Performing KFold cross validation -----"
print "---------------------------------------------"

from sklearn.model_selection import KFold

kf = KFold(n_splits=4,shuffle=True,random_state=10)
print kf

for train_index, test_index in kf.split(features):
    print "Train:\n",train_index
    print "Test:\n",test_index
    print "----------"
    features_train = [features[ii] for ii in train_index]
    features_test = [features[ii] for ii in test_index]
    labels_train = [labels[ii] for ii in train_index]
    labels_test = [labels[ii] for ii in test_index]

    from sklearn import tree
    clf = tree.DecisionTreeClassifier()

    t0 = time()
    clf.fit(features_train,labels_train)
    print "Training time: ",round(time()-t0,3),"s"


    t1= time()
    pred = clf.predict(features_test)
    print "Predicting time: ",round(time()-t1,3),"s"

    print "Accuracy= ", round(clf.score(features_test,labels_test),2)

    print classification_report(labels_test,pred)

print "--------------------------------------------------------"
print "----- Using best split from KFold Cross Validation -----"
print "--------------------------------------------------------"

train_index = [0,1 ,2 ,4 ,5 ,7 ,8 ,9 ,11,12,13,15,16,17,18,19,21,
  22,23,24,25,26,27,28,30,31,33,35,36,38,39,40,41,44,46,
  47,48,49,50,51,52,53,54,55,57,59,62,63,64,65,66,67,69,
  70,71,72,73,74,75,77,78,79,80,81,82,84,85,86,88,89,90,
  91,92,93,97,98,99,101,102,105,106,107,108,109,111,113,114,115,
 120,121,123,124,125,126,127,129,133,134,135,136,137,138,139,140]

test_index = [3,6,14,20,29,32,34,37,42,43,45,56,58,60,61,68,76,83,
  87,94,95,96,103,104,110,112,116,117,118,119,122,128,130,131,132]

features_train = [features[ii] for ii in train_index]
features_test = [features[ii] for ii in test_index]
labels_train = [labels[ii] for ii in train_index]
labels_test = [labels[ii] for ii in test_index]

print "------------------------------------------"
print "----- After tuning with GridSearchCV -----"
print "------------------------------------------"

parameters_dtc = {'criterion' : ['gini','entropy'],
                'min_samples_split' : [2,3,4,5,6],
                'presort' : [False,True]
            }

print "----- Decision Tree -----"

dtc = tree.DecisionTreeClassifier(random_state=42)
clf_dtc = GridSearchCV(dtc,parameters_dtc)

t0 = time()
clf_dtc.fit(features_train,labels_train)
print "Training time: ",round(time()-t0,3),"s"


t1= time()
pred = clf_dtc.predict(features_test)
print "Predicting time: ",round(time()-t1,3),"s"

print "Accuracy= ", round(clf_dtc.score(features_test,labels_test),2)

print classification_report(labels_test,pred)

print "Best parameters: ", clf_dtc.best_params_

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf_dtc, my_dataset, features_list)