import sys
import pickle
import math
import pprint
from time import time
from sklearn import cross_validation

from sklearn.metrics import accuracy_score, classification_report

from feature_format import featureFormat, targetFeatureSplit

complete_features_list = ['poi',
                'salary',
                'bonus',
                'deferral_payments',
                'deferred_income',
                'director_fees',
                'expenses',
                'from_messages',
                'to_messages',
                'loan_advances',
                'long_term_incentive',
                'other',
                'restricted_stock',
                'restricted_stock_deferred',
                'shared_receipt_with_poi',
                'total_payments',
                'exercised_stock_options',
                'from_poi_to_this_person',
                'from_this_person_to_poi',
                'total_stock_value',
                'percent_from_poi_to_this_person',
                'percent_from_this_person_to_poi']

"""
Looking at the number of NaN values for the various features, the feature 
director_fees has 129 NaN values out of 146 data points. 
Similarly, loan_advances has 142 NaN values and restricted_stock_deferred has 120 
NaN values.

Hence, these three features will not be included in any of the feature combinations 
below.
"""


features_used = {
    'all' : ['poi',
                'salary',
                'bonus',
                'deferral_payments',
                'deferred_income',
                'expenses',
                'from_messages',
                'to_messages',
                'long_term_incentive',
                'other',
                'restricted_stock',
                'shared_receipt_with_poi',
                'total_payments',
                'exercised_stock_options',
                'from_poi_to_this_person',
                'from_this_person_to_poi',
                'total_stock_value',
                'percent_from_poi_to_this_person',
                'percent_from_this_person_to_poi'],
    'basic_financial' : ['poi',
                'salary',
                'bonus'],
    'all_financial' : ['poi',
                'salary',
                'bonus',
                'deferral_payments',
                'deferred_income',
                'expenses',
                'long_term_incentive',
                'other',
                'restricted_stock',
                'shared_receipt_with_poi',
                'total_payments',
                'exercised_stock_options',
                'total_stock_value'],
    'basic_email' : ['poi',
                'from_poi_to_this_person',
                'from_this_person_to_poi',
                'percent_from_poi_to_this_person',
                'percent_from_this_person_to_poi'],
    'all_email' : ['poi',
                'from_messages',
                'to_messages',
                'from_poi_to_this_person',
                'from_this_person_to_poi',
                'percent_from_poi_to_this_person',
                'percent_from_this_person_to_poi'],
    'mixed_1' : ['poi',
                'salary',
                'bonus',
                'percent_from_poi_to_this_person',
                'percent_from_this_person_to_poi'],
    'mixed_2' : ['poi',
                'salary',
                'bonus',
                'from_poi_to_this_person',
                'from_this_person_to_poi'],
    'mixed_3' : ['poi',
                'salary',
                'bonus',
                'exercised_stock_options',
                'total_stock_value'],
    'mixed_4' : ['poi',
                'salary',
                'bonus',
                'exercised_stock_options',
                'total_stock_value',
                'from_poi_to_this_person',
                'from_this_person_to_poi',
                'expenses',
                'total_payments']
}

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Removing outliers
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

for key,features_list in features_used.items():

    print "\n---------- ",key," ----------"
    print "Features used:"
    print features_list

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
                        max_val = int(value[feature])
                        min_val = int(value[feature])
                        first = False
                    else:
                        if value[feature] > max_val:
                            max_val = int(value[feature])
                        if value[feature] < min_val:
                            min_val = int(value[feature])
        
            for key,value in data_dict.items():
                if not math.isnan(float(value[feature])):
                    value[feature] = (featureScaling(value[feature],max_val,min_val))

    ### Extract features and labels from dataset for local testing
    data = featureFormat(data_dict, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    features_train,features_test,labels_train,labels_test = \
        cross_validation.train_test_split(features,labels,test_size=0.3,random_state=42)


    ### Creating and testing performance of classifier
    from sklearn.svm import SVC
    clf = SVC()

    t0 = time()
    clf.fit(features_train,labels_train)
    print "Training time: ",round(time()-t0,3),"s"


    t1= time()
    pred = clf.predict(features_test)
    print "Predicting time: ",round(time()-t1,3),"s"

    print "Accuracy = ",clf.score(features_test,labels_test)

    print classification_report(labels_test,pred)

"""    
As we can see from the performance of the different features, the feature groups: 
basic_financial, mixed_2, mixed_3 and mixed_4 perform well across accuracy, precision,
and recall.

Hence, these four feature groups will be further tested in the POI identifier code (poi_id.py)
using the tester.py script for different classifiers.

How the performace of the classifier is affected in by the new features created is 
investigated in new_feature_performance.py.
"""