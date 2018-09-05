import sys
import pickle
import math
import pprint
from time import time
from sklearn import cross_validation

from sklearn.metrics import accuracy_score, classification_report

from feature_format import featureFormat, targetFeatureSplit

features_used = {
    'basic_financial' : ['poi',
                'salary',
                'bonus'],
    'basic_financial_with_new_features' : ['poi',
                'salary',
                'bonus',
                'percent_from_poi_to_this_person',
                'percent_from_this_person_to_poi']
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

    print "\n---------- ",key," ----------\n"

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
From the performance metrics we can see that the addition of the new features 
negatively affects the performance of the classifier.

As such, these new features will not be used in the final POI identifier (poi_id.py)
"""