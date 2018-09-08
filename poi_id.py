import sys
import pickle
import math
import pprint
from time import time
from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.metrics import accuracy_score, classification_report


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

features_list = ['poi',
                'salary',
                'bonus',
                'exercised_stock_options',
                'total_stock_value',
                'from_poi_to_this_person',
                'from_this_person_to_poi',
                'expenses',
                'total_payments']

"""
After testing the four feature groups from the results of feature_selection.py. The
mixed_4 feature group gives the best performance when tested with tester.py.

For full details about the feature selection, consult feature_selection.py and
new_feature_performance.py.
"""

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


data_dict.pop('TOTAL',0) # Full outlier exploration in identify_outliers.py

### Creating new feature(s)
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

features_train,features_test,labels_train,labels_test = \
        train_test_split(features,labels,test_size=0.3,random_state=42)
#features_train,features_test,labels_train,labels_test = \
    #cross_validation.train_test_split(features,labels,test_size=0.3,random_state=42)

# Trying a variety of classifiers

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
overall the decision tree classifier performs better considering precision, recall
and f1-score.
"""

# Dumping the classifier, dataset and features_list so that anyone can check the 
# results

dump_classifier_and_data(clf_dtc, my_dataset, features_list)